"""Minimal FastAPI API for Milestone 8A controller development."""

from __future__ import annotations

import tempfile
import json
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from psycopg.types.json import Jsonb

from controller_common.config import default_r2_bucket
from controller_common.db import (
    cancel_stage_run,
    connect,
    create_stage_run as insert_stage_run,
    enqueue_next_stage_after_approval,
    ensure_schema,
    new_id,
    reject_stage_run,
    retry_stage_run,
    row_to_json,
    rows_to_json,
    stage_queued_status,
)
from controller_common.raw_upload import (
    manifest_summary,
    metadata_overrides_by_filename,
    upload_raw_directory,
    uploaded_file_names,
    write_upload_file,
)
from controller_ui.routes import router as ui_router


StageName = Literal["preprocess", "colmap", "training"]


class ProjectCreate(BaseModel):
    id: Optional[str] = None
    name: str
    raw_uri: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    raw_uri: Optional[str] = None
    preprocess_current_uri: Optional[str] = None
    colmap_current_uri: Optional[str] = None
    training_current_uri: Optional[str] = None


class StageRunCreate(BaseModel):
    id: Optional[str] = None
    stage: StageName
    status: Optional[str] = None
    attempt: int = Field(default=1, ge=1)
    image: Optional[str] = None
    provider: str = "local_fake"
    command: Optional[str] = None
    input_uri_json: dict[str, Any] = Field(default_factory=dict)
    output_uri: Optional[str] = None
    summary_json: dict[str, Any] = Field(default_factory=dict)
    progress_json: dict[str, Any] = Field(default_factory=dict)


class StageRunUpdate(BaseModel):
    status: Optional[str] = None
    image: Optional[str] = None
    provider: Optional[str] = None
    provider_job_id: Optional[str] = None
    provider_pod_id: Optional[str] = None
    command: Optional[str] = None
    input_uri_json: Optional[dict[str, Any]] = None
    output_uri: Optional[str] = None
    summary_json: Optional[dict[str, Any]] = None
    progress_json: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None


class PreprocessQueueRequest(BaseModel):
    raw_uri: Optional[str] = None
    output_uri: Optional[str] = None
    endpoint_url: Optional[str] = None
    profile: str = "indoor_room"
    python_bin: Optional[str] = None
    preprocess_args: list[str] = Field(default_factory=list)
    provider: str = "local_preprocess"
    dry_run: bool = False


class ColmapQueueRequest(BaseModel):
    preprocess_uri: Optional[str] = None
    output_uri: Optional[str] = None
    endpoint_url: Optional[str] = None
    mode: str = "global"
    matcher: str = "exhaustive"
    camera_model: str = "SIMPLE_RADIAL"
    provider: str = "runpod_colmap"
    image: Optional[str] = None
    repo_url: Optional[str] = None
    git_ref: Optional[str] = None
    colmap_args: list[str] = Field(default_factory=list)
    dry_run: bool = False


class ApprovalCreate(BaseModel):
    stage: StageName
    stage_run_id: Optional[str] = None
    notes: Optional[str] = None


class ApprovalDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    notes: Optional[str] = None


class StageActionRequest(BaseModel):
    notes: Optional[str] = None


app = FastAPI(title="Buildvision3D Controller API", version="0.1.0")
app.mount("/ui/static", StaticFiles(directory="controller_ui/static"), name="controller-ui-static")
app.include_router(ui_router, prefix="/ui")


@app.on_event("startup")
def startup() -> None:
    ensure_schema()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/projects")
def list_projects() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
    return rows_to_json(rows)


@app.post("/projects", status_code=201)
def create_project(payload: ProjectCreate) -> dict[str, Any]:
    project_id = payload.id or new_id("project")
    status = "raw_uploaded" if payload.raw_uri else "created"
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO projects (id, name, status, raw_uri)
            VALUES (%s, %s, %s, %s)
            RETURNING *
            """,
            (project_id, payload.name, status, payload.raw_uri),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create project")
    return row_to_json(row)


@app.get("/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return row_to_json(row)


@app.patch("/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate) -> dict[str, Any]:
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return get_project(project_id)

    allowed_fields = {
        "name",
        "status",
        "raw_uri",
        "preprocess_current_uri",
        "colmap_current_uri",
        "training_current_uri",
    }
    set_parts = [f"{field} = %s" for field in updates if field in allowed_fields]
    values = [updates[field] for field in updates if field in allowed_fields]
    if not set_parts:
        return get_project(project_id)
    values.append(project_id)
    with connect() as conn:
        row = conn.execute(
            f"""
            UPDATE projects
            SET {', '.join(set_parts)}, updated_at = now()
            WHERE id = %s
            RETURNING *
            """,
            values,
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return row_to_json(row)


@app.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str) -> None:
    with connect() as conn:
        result = conn.execute("DELETE FROM projects WHERE id = %s", (project_id,))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Project not found")


@app.post("/projects/{project_id}/raw", status_code=201)
def upload_project_raw(
    project_id: str,
    files: list[UploadFile] = File(...),
    name: Optional[str] = Form(default=None),
    destination_uri: Optional[str] = Form(default=None),
    endpoint_url: Optional[str] = Form(default=None),
    metadata_json: Optional[str] = Form(default=None),
    delete: bool = Form(default=False),
    dry_run: bool = Form(default=False),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    raw_uri = destination_uri or f"r2://{default_r2_bucket()}/projects/{project_id}/raw"
    require_r2_uri(raw_uri, "destination_uri")
    metadata = parse_metadata_json(metadata_json)
    try:
        metadata_overrides = metadata_overrides_by_filename(metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix=f"buildvision3d-api-upload-{project_id}-") as temp_dir:
        upload_root = Path(temp_dir) / "raw"
        upload_root.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        try:
            for upload in files:
                saved_paths.append(write_upload_file(upload_root, upload.filename or "upload.bin", upload.file))
            manifest = upload_raw_directory(
                project_id=project_id,
                input_dir=upload_root,
                destination_uri=raw_uri,
                endpoint_url=endpoint_url,
                metadata_overrides=metadata_overrides,
                delete=delete,
                dry_run=dry_run,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Raw upload failed: {exc}") from exc

        project_name = name or project_id
        row = None
        if not dry_run:
            with connect() as conn:
                project = conn.execute("SELECT id FROM projects WHERE id = %s", (project_id,)).fetchone()
                if project is None:
                    row = conn.execute(
                        """
                        INSERT INTO projects (id, name, status, raw_uri)
                        VALUES (%s, %s, 'raw_uploaded', %s)
                        RETURNING *
                        """,
                        (project_id, project_name, raw_uri),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        UPDATE projects
                        SET name = COALESCE(%s, name),
                            status = 'raw_uploaded',
                            raw_uri = %s,
                            updated_at = now()
                        WHERE id = %s
                        RETURNING *
                        """,
                        (name, raw_uri, project_id),
                    ).fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Raw upload finished but project update failed")
        return {
            "project": row_to_json(row) if row is not None else None,
            "raw_uri": raw_uri,
            "dry_run": dry_run,
            "uploaded_files": uploaded_file_names(saved_paths, upload_root),
            "manifest_summary": manifest_summary(manifest),
            "sources": manifest.get("sources", []),
        }


def parse_metadata_json(raw_value: Optional[str]) -> Optional[dict[str, Any]]:
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata_json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata_json must be a JSON object")
    return parsed


@app.get("/stage-runs")
def list_stage_runs(project_id: Optional[str] = None) -> list[dict[str, Any]]:
    with connect() as conn:
        if project_id:
            rows = conn.execute(
                "SELECT * FROM stage_runs WHERE project_id = %s ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM stage_runs ORDER BY created_at DESC").fetchall()
    return rows_to_json(rows)


@app.post("/projects/{project_id}/stage-runs", status_code=201)
def create_stage_run(project_id: str, payload: StageRunCreate) -> dict[str, Any]:
    stage_run_id = payload.id or new_id(f"{payload.stage}_run")
    status = payload.status or stage_queued_status(payload.stage)
    with connect() as conn:
        project = conn.execute("SELECT id FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        row = insert_stage_run(
            conn,
            project_id=project_id,
            stage=payload.stage,
            status=status,
            attempt=payload.attempt,
            image=payload.image,
            provider=payload.provider,
            command=payload.command,
            input_uri_json=payload.input_uri_json,
            output_uri=payload.output_uri,
            summary_json=payload.summary_json,
            progress_json=payload.progress_json,
            stage_run_id=stage_run_id,
        )
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create stage run")
    return row_to_json(row)


@app.post("/projects/{project_id}/preprocess", status_code=201)
def queue_preprocess(project_id: str, payload: PreprocessQueueRequest) -> dict[str, Any]:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        raw_uri = payload.raw_uri or project.get("raw_uri")
        if not raw_uri:
            raise HTTPException(status_code=400, detail="Project raw_uri is required to queue preprocessing")
        output_uri = payload.output_uri or f"r2://{default_r2_bucket()}/projects/{project_id}/preprocess"
        require_r2_uri(raw_uri, "raw_uri")
        require_r2_uri(output_uri, "output_uri")
        input_uri_json = {
            "raw_uri": raw_uri,
            "output_uri": output_uri,
            "endpoint_url": payload.endpoint_url,
            "profile": payload.profile,
            "python_bin": payload.python_bin,
            "preprocess_args": payload.preprocess_args,
            "dry_run": payload.dry_run,
        }
        row = insert_stage_run(
            conn,
            project_id=project_id,
            stage="preprocess",
            provider=payload.provider,
            input_uri_json={key: value for key, value in input_uri_json.items() if value not in (None, [])},
            output_uri=f"{output_uri.rstrip('/')}/current",
        )
    return row_to_json(row)


@app.post("/projects/{project_id}/colmap", status_code=201)
def queue_colmap(project_id: str, payload: ColmapQueueRequest) -> dict[str, Any]:
    with connect() as conn:
        project = conn.execute("SELECT * FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        preprocess_uri = payload.preprocess_uri or project.get("preprocess_current_uri")
        if not preprocess_uri:
            raise HTTPException(status_code=400, detail="Project preprocess_current_uri is required to queue COLMAP")
        output_uri = payload.output_uri or f"r2://{default_r2_bucket()}/projects/{project_id}/colmap"
        require_r2_uri(preprocess_uri, "preprocess_uri")
        require_r2_uri(output_uri, "output_uri")
        input_uri_json = {
            "preprocess_uri": preprocess_uri,
            "output_uri": output_uri,
            "endpoint_url": payload.endpoint_url,
            "mode": payload.mode,
            "matcher": payload.matcher,
            "camera_model": payload.camera_model,
            "repo_url": payload.repo_url,
            "git_ref": payload.git_ref,
            "colmap_args": payload.colmap_args,
            "dry_run": payload.dry_run,
        }
        row = insert_stage_run(
            conn,
            project_id=project_id,
            stage="colmap",
            provider=payload.provider,
            image=payload.image,
            input_uri_json={key: value for key, value in input_uri_json.items() if value not in (None, [])},
            output_uri=f"{output_uri.rstrip('/')}/current",
        )
    return row_to_json(row)


def require_r2_uri(value: str, field_name: str) -> None:
    if not value.startswith("r2://"):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be an r2:// URI for real preprocessing",
        )


@app.get("/stage-runs/{stage_run_id}")
def get_stage_run(stage_run_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Stage run not found")
    return row_to_json(row)


@app.patch("/stage-runs/{stage_run_id}")
def update_stage_run(stage_run_id: str, payload: StageRunUpdate) -> dict[str, Any]:
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return get_stage_run(stage_run_id)

    json_fields = {"input_uri_json", "summary_json", "progress_json"}
    allowed_fields = {
        "status",
        "image",
        "provider",
        "provider_job_id",
        "provider_pod_id",
        "command",
        "input_uri_json",
        "output_uri",
        "summary_json",
        "progress_json",
        "error_message",
    }
    set_parts = []
    values = []
    for field, value in updates.items():
        if field not in allowed_fields:
            continue
        set_parts.append(f"{field} = %s")
        values.append(Jsonb(value) if field in json_fields else value)
    if not set_parts:
        return get_stage_run(stage_run_id)
    values.append(stage_run_id)
    with connect() as conn:
        row = conn.execute(
            f"""
            UPDATE stage_runs
            SET {', '.join(set_parts)}, updated_at = now()
            WHERE id = %s
            RETURNING *
            """,
            values,
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Stage run not found")
    return row_to_json(row)


@app.delete("/stage-runs/{stage_run_id}", status_code=204)
def delete_stage_run(stage_run_id: str) -> None:
    with connect() as conn:
        result = conn.execute("DELETE FROM stage_runs WHERE id = %s", (stage_run_id,))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Stage run not found")


@app.post("/stage-runs/{stage_run_id}/approve", status_code=201)
def approve_stage_run(stage_run_id: str, payload: StageActionRequest = StageActionRequest()) -> dict[str, Any]:
    with connect() as conn:
        with conn.transaction():
            try:
                row = enqueue_next_stage_after_approval(conn, stage_run_id=stage_run_id, notes=payload.notes)
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    return row_to_json(row)


@app.post("/stage-runs/{stage_run_id}/reject")
def reject_stage_run_endpoint(stage_run_id: str, payload: StageActionRequest = StageActionRequest()) -> dict[str, Any]:
    with connect() as conn:
        with conn.transaction():
            try:
                row = reject_stage_run(conn, stage_run_id=stage_run_id, notes=payload.notes)
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    return row_to_json(row)


@app.post("/stage-runs/{stage_run_id}/retry", status_code=201)
def retry_stage_run_endpoint(stage_run_id: str, payload: StageActionRequest = StageActionRequest()) -> dict[str, Any]:
    with connect() as conn:
        with conn.transaction():
            try:
                row = retry_stage_run(conn, stage_run_id=stage_run_id, notes=payload.notes)
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
    return row_to_json(row)


@app.post("/stage-runs/{stage_run_id}/cancel")
def cancel_stage_run_endpoint(stage_run_id: str, payload: StageActionRequest = StageActionRequest()) -> dict[str, Any]:
    with connect() as conn:
        with conn.transaction():
            try:
                row = cancel_stage_run(conn, stage_run_id=stage_run_id, notes=payload.notes)
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
    return row_to_json(row)


@app.get("/stage-runs/{stage_run_id}/events")
def list_stage_run_events(stage_run_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        stage_run = conn.execute("SELECT id FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
        if stage_run is None:
            raise HTTPException(status_code=404, detail="Stage run not found")
        rows = conn.execute(
            """
            SELECT *
            FROM events
            WHERE stage_run_id = %s
            ORDER BY created_at ASC
            """,
            (stage_run_id,),
        ).fetchall()
    return rows_to_json(rows)


@app.get("/approvals")
def list_approvals(project_id: Optional[str] = None) -> list[dict[str, Any]]:
    with connect() as conn:
        if project_id:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE project_id = %s ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM approvals ORDER BY created_at DESC").fetchall()
    return rows_to_json(rows)


@app.post("/projects/{project_id}/approvals", status_code=201)
def create_approval(project_id: str, payload: ApprovalCreate) -> dict[str, Any]:
    approval_id = new_id("approval")
    with connect() as conn:
        project = conn.execute("SELECT id FROM projects WHERE id = %s", (project_id,)).fetchone()
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        row = conn.execute(
            """
            INSERT INTO approvals (id, project_id, stage, stage_run_id, notes)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (approval_id, project_id, payload.stage, payload.stage_run_id, payload.notes),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create approval")
    return row_to_json(row)


@app.post("/approvals/{approval_id}/decision")
def decide_approval(approval_id: str, payload: ApprovalDecision) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute(
            """
            UPDATE approvals
            SET status = 'decided',
                decision = %s,
                notes = COALESCE(%s, notes),
                decided_at = now()
            WHERE id = %s
            RETURNING *
            """,
            (payload.decision, payload.notes, approval_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return row_to_json(row)
