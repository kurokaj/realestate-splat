"""Postgres access helpers for the controller skeleton."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from controller_common.config import database_url, default_r2_bucket


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


QUEUED_STATUSES = ("preprocess_queued", "colmap_queued", "training_queued")
TERMINAL_STATUSES = ("approved", "completed", "failed", "cancelled")
REJECTED_STATUS_BY_STAGE = {
    "preprocess": "preprocess_rejected",
    "colmap": "colmap_rejected",
}
RUNNING_STATUS_BY_STAGE = {
    "preprocess": "preprocess_running",
    "colmap": "colmap_running",
    "training": "training_running",
}
AWAITING_APPROVAL_STATUS_BY_STAGE = {
    "preprocess": "awaiting_preprocess_approval",
    "colmap": "awaiting_colmap_approval",
}
ACTIVE_RUN_FIELD_BY_STAGE = {
    "preprocess": "active_preprocess_run_id",
    "colmap": "active_colmap_run_id",
    "training": "active_training_run_id",
}
CURRENT_URI_FIELD_BY_STAGE = {
    "preprocess": "preprocess_current_uri",
    "colmap": "colmap_current_uri",
    "training": "training_current_uri",
}


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    with psycopg.connect(database_url(), row_factory=dict_row) as conn:
        yield conn


def ensure_schema() -> None:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.execute(schema)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def normalize_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    return value


def row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "isoformat"):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def rows_to_json(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row_to_json(row) for row in rows]


def stage_queued_status(stage: str) -> str:
    return f"{stage}_queued"


def default_current_uri(project_id: str, stage: str) -> str:
    return f"r2://{default_r2_bucket()}/projects/{project_id}/{stage}/current"


def default_stage_output_uri(project_id: str, stage: str) -> str:
    return f"r2://{default_r2_bucket()}/projects/{project_id}/{stage}"


def stage_prefix(stage: str) -> str:
    return {
        "preprocess": "preprocess",
        "colmap": "colmap",
        "training": "training",
    }.get(stage, "stage")


def create_event(
    conn: psycopg.Connection,
    *,
    stage_run_id: str,
    kind: str,
    message: str,
    level: str = "info",
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    event_id = new_id("event")
    row = conn.execute(
        """
        INSERT INTO events (id, stage_run_id, level, kind, message, payload_json)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING *
        """,
        (event_id, stage_run_id, level, kind, message, Jsonb(payload or {})),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to create event")
    return row


def claim_next_queued_stage(conn: psycopg.Connection, *, claimed_by: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        """
        WITH candidate AS (
            SELECT id
            FROM stage_runs
            WHERE status = ANY(%s)
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        UPDATE stage_runs AS stage_run
        SET status = CASE stage_run.stage
                WHEN 'preprocess' THEN 'preprocess_running'
                WHEN 'colmap' THEN 'colmap_running'
                WHEN 'training' THEN 'training_running'
                ELSE 'failed'
            END,
            claimed_by = %s,
            claimed_at = now(),
            started_at = COALESCE(stage_run.started_at, now()),
            updated_at = now(),
            progress_json = jsonb_build_object('percent', 0, 'message', 'Claimed by worker')
        FROM candidate
        WHERE stage_run.id = candidate.id
        RETURNING stage_run.*
        """,
        (list(QUEUED_STATUSES), claimed_by),
    ).fetchone()
    if row is not None:
        create_event(
            conn,
            stage_run_id=row["id"],
            kind="stage_claimed",
            message=f"Claimed {row['stage']} stage",
            payload={"claimed_by": claimed_by},
        )
    return row


def complete_stage_run(
    conn: psycopg.Connection,
    *,
    stage_run_id: str,
    summary: dict[str, Any],
    output_uri: Optional[str],
) -> dict[str, Any]:
    stage_row = conn.execute("SELECT * FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
    if stage_row is None:
        raise LookupError(f"Stage run not found: {stage_run_id}")

    stage = stage_row["stage"]
    final_status = AWAITING_APPROVAL_STATUS_BY_STAGE.get(stage, "completed")
    updated = conn.execute(
        """
        UPDATE stage_runs
        SET status = %s,
            summary_json = %s,
            progress_json = jsonb_build_object('percent', 100, 'message', 'Stage complete'),
            output_uri = COALESCE(%s, output_uri),
            error_message = NULL,
            finished_at = now(),
            updated_at = now()
        WHERE id = %s
        RETURNING *
        """,
        (final_status, Jsonb(summary), output_uri, stage_run_id),
    ).fetchone()
    if updated is None:
        raise RuntimeError(f"Failed to complete stage run: {stage_run_id}")

    project_updates = ["status = %s", "updated_at = now()"]
    project_values: list[Any] = [final_status]
    active_field = ACTIVE_RUN_FIELD_BY_STAGE.get(stage)
    current_uri_field = CURRENT_URI_FIELD_BY_STAGE.get(stage)
    input_json = stage_row.get("input_uri_json") if isinstance(stage_row.get("input_uri_json"), dict) else {}
    is_partial_preprocess = stage == "preprocess" and input_json.get("preprocess_scope") == "group"
    if active_field:
        project_updates.append(f"{active_field} = %s")
        project_values.append(stage_run_id)
    if current_uri_field and output_uri and not is_partial_preprocess:
        project_updates.append(f"{current_uri_field} = %s")
        project_values.append(output_uri)
    project_values.append(updated["project_id"])
    conn.execute(
        f"UPDATE projects SET {', '.join(project_updates)} WHERE id = %s",
        project_values,
    )
    create_event(
        conn,
        stage_run_id=stage_run_id,
        kind="stage_completed",
        message=f"{stage} stage completed",
        payload={"output_uri": output_uri},
    )
    return updated


def create_stage_run(
    conn: psycopg.Connection,
    *,
    project_id: str,
    stage: str,
    status: Optional[str] = None,
    attempt: int = 1,
    image: Optional[str] = None,
    provider: str = "local_fake",
    command: Optional[str] = None,
    input_uri_json: Optional[dict[str, Any]] = None,
    output_uri: Optional[str] = None,
    summary_json: Optional[dict[str, Any]] = None,
    progress_json: Optional[dict[str, Any]] = None,
    stage_run_id: Optional[str] = None,
) -> dict[str, Any]:
    row = conn.execute(
        """
        INSERT INTO stage_runs (
            id, project_id, stage, status, attempt, image, provider, command,
            input_uri_json, output_uri, summary_json, progress_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """,
        (
            stage_run_id or new_id(f"{stage}_run"),
            project_id,
            stage,
            status or stage_queued_status(stage),
            attempt,
            image,
            provider,
            command,
            Jsonb(input_uri_json or {}),
            output_uri,
            Jsonb(summary_json or {}),
            Jsonb(progress_json or {}),
        ),
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to create stage run")
    conn.execute(
        "UPDATE projects SET status = %s, updated_at = now() WHERE id = %s",
        (row["status"], project_id),
    )
    return row


def enqueue_next_stage_after_approval(
    conn: psycopg.Connection,
    *,
    stage_run_id: str,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    stage_run = conn.execute("SELECT * FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
    if stage_run is None:
        raise LookupError(f"Stage run not found: {stage_run_id}")
    stage = stage_run["stage"]
    if stage == "preprocess":
        if stage_run["status"] != "awaiting_preprocess_approval":
            raise ValueError("Preprocess stage is not awaiting approval")
    elif stage == "colmap":
        if stage_run["status"] != "awaiting_colmap_approval":
            raise ValueError("COLMAP stage is not awaiting approval")
    else:
        raise ValueError(f"Stage does not have an approval transition: {stage}")

    approval_id = new_id("approval")
    conn.execute(
        """
        INSERT INTO approvals (id, project_id, stage, stage_run_id, status, decision, notes, decided_at)
        VALUES (%s, %s, %s, %s, 'decided', 'approved', %s, now())
        """,
        (approval_id, stage_run["project_id"], stage, stage_run_id, notes),
    )
    create_event(
        conn,
        stage_run_id=stage_run_id,
        kind="stage_approved",
        message=f"Approved {stage} stage",
        payload={"notes": notes},
    )
    row = conn.execute(
        """
        UPDATE stage_runs
        SET status = 'approved',
            updated_at = now()
        WHERE id = %s
        RETURNING *
        """,
        (stage_run_id,),
    ).fetchone()
    conn.execute(
        """
        UPDATE projects
        SET status = 'approved',
            updated_at = now()
        WHERE id = %s
        """,
        (stage_run["project_id"],),
    )
    if row is None:
        raise RuntimeError(f"Failed to approve stage run: {stage_run_id}")
    return row


def reject_stage_run(conn: psycopg.Connection, *, stage_run_id: str, notes: Optional[str] = None) -> dict[str, Any]:
    stage_run = conn.execute("SELECT * FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
    if stage_run is None:
        raise LookupError(f"Stage run not found: {stage_run_id}")
    rejected_status = REJECTED_STATUS_BY_STAGE.get(stage_run["stage"])
    if rejected_status is None:
        raise ValueError(f"Stage cannot be rejected: {stage_run['stage']}")
    expected_status = AWAITING_APPROVAL_STATUS_BY_STAGE.get(stage_run["stage"])
    if stage_run["status"] != expected_status:
        raise ValueError(f"Stage is not awaiting approval: {stage_run['status']}")
    row = conn.execute(
        """
        UPDATE stage_runs
        SET status = %s,
            error_message = COALESCE(%s, error_message),
            updated_at = now()
        WHERE id = %s
        RETURNING *
        """,
        (rejected_status, notes, stage_run_id),
    ).fetchone()
    approval_id = new_id("approval")
    conn.execute(
        """
        INSERT INTO approvals (id, project_id, stage, stage_run_id, status, decision, notes, decided_at)
        VALUES (%s, %s, %s, %s, 'decided', 'rejected', %s, now())
        """,
        (approval_id, stage_run["project_id"], stage_run["stage"], stage_run_id, notes),
    )
    conn.execute(
        "UPDATE projects SET status = %s, updated_at = now() WHERE id = %s",
        (rejected_status, stage_run["project_id"]),
    )
    create_event(
        conn,
        stage_run_id=stage_run_id,
        kind="stage_rejected",
        message=f"Rejected {stage_run['stage']} stage",
        payload={"notes": notes},
    )
    return row


def cancel_stage_run(conn: psycopg.Connection, *, stage_run_id: str, notes: Optional[str] = None) -> dict[str, Any]:
    row = conn.execute(
        """
        UPDATE stage_runs
        SET status = 'cancelled',
            error_message = COALESCE(%s, error_message),
            finished_at = COALESCE(finished_at, now()),
            updated_at = now()
        WHERE id = %s
          AND status <> ALL(%s)
        RETURNING *
        """,
        (notes, stage_run_id, list(TERMINAL_STATUSES)),
    ).fetchone()
    if row is None:
        raise LookupError(f"Cancellable stage run not found: {stage_run_id}")
    conn.execute(
        "UPDATE projects SET status = 'cancelled', updated_at = now() WHERE id = %s",
        (row["project_id"],),
    )
    create_event(
        conn,
        stage_run_id=stage_run_id,
        kind="stage_cancelled",
        message=f"Cancelled {row['stage']} stage",
        payload={"notes": notes},
    )
    return row


def retry_stage_run(conn: psycopg.Connection, *, stage_run_id: str, notes: Optional[str] = None) -> dict[str, Any]:
    stage_run = conn.execute("SELECT * FROM stage_runs WHERE id = %s", (stage_run_id,)).fetchone()
    if stage_run is None:
        raise LookupError(f"Stage run not found: {stage_run_id}")
    create_event(
        conn,
        stage_run_id=stage_run_id,
        kind="stage_retry_requested",
        message=f"Retry requested for {stage_run['stage']} stage",
        payload={"notes": notes},
    )
    return create_stage_run(
        conn,
        project_id=stage_run["project_id"],
        stage=stage_run["stage"],
        attempt=int(stage_run["attempt"] or 1) + 1,
        image=stage_run["image"],
        provider=stage_run["provider"],
        command=stage_run["command"],
        input_uri_json=stage_run["input_uri_json"],
        output_uri=stage_run["output_uri"],
        summary_json={},
        progress_json={"percent": 0, "message": "Retry queued"},
    )
