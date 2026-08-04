CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    raw_uri TEXT,
    preprocess_current_uri TEXT,
    colmap_current_uri TEXT,
    training_current_uri TEXT,
    active_preprocess_run_id TEXT,
    active_colmap_run_id TEXT,
    active_training_run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stage_runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    image TEXT,
    provider TEXT NOT NULL DEFAULT 'local_fake',
    provider_job_id TEXT,
    provider_pod_id TEXT,
    command TEXT,
    input_uri_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_uri TEXT,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    progress_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_stage_runs_project_id ON stage_runs(project_id);
CREATE INDEX IF NOT EXISTS idx_stage_runs_status_created_at ON stage_runs(status, created_at);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    stage_run_id TEXT NOT NULL REFERENCES stage_runs(id) ON DELETE CASCADE,
    level TEXT NOT NULL DEFAULT 'info',
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_stage_run_id_created_at ON events(stage_run_id, created_at);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    stage_run_id TEXT REFERENCES stage_runs(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    decision TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_approvals_project_stage ON approvals(project_id, stage);
