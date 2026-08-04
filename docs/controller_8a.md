# Milestone 8A Local Controller Skeleton

Milestone 8A proves that Postgres state can drive the pipeline graph before any
real GPU provider integration exists.

## Runtime Shape

The controller can run either as a terminal-first development loop or as a
single Compose stack.

Terminal-first:

```bash
docker compose up postgres
uvicorn controller_api.main:app --reload --host 0.0.0.0 --port 8000
python -m controller_worker
```

Compose:

```bash
docker compose up --build
```

The API and worker both read `DATABASE_URL`. The default local value is:

```text
postgresql://buildvision3d:buildvision3d@localhost:5432/buildvision3d
```

Copy `.env.example` to `.env` if you want shell tooling to load the same
settings. Do not commit real R2 or provider credentials. Compose reads
environment variables from your shell or `.env` for credentials, but it
overrides `DATABASE_URL` and `CONTROLLER_STAGE_PYTHON_BIN` inside containers so
they use the Postgres service name and container Python.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Smoke Test

### Terminal-First

Start Postgres:

```bash
docker compose up postgres
```

In another terminal, start the API:

```bash
uvicorn controller_api.main:app --reload --host 0.0.0.0 --port 8000
```

Create a project:

```bash
curl -s -X POST http://localhost:8000/projects \
  -H 'content-type: application/json' \
  -d '{"id":"car_single_smoke","name":"Car single smoke","raw_uri":"r2://buildvision3d-pipeline/projects/car_single_smoke/raw"}'
```

Enqueue a fake COLMAP stage:

```bash
curl -s -X POST http://localhost:8000/projects/car_single_smoke/stage-runs \
  -H 'content-type: application/json' \
  -d '{"stage":"colmap","provider":"local_fake","input_uri_json":{"preprocess":"r2://buildvision3d-pipeline/projects/car_single_smoke/preprocess/current"}}'
```

In a third terminal, process one queued stage:

```bash
python -m controller_worker --once
```

Inspect stage runs and events:

```bash
curl -s http://localhost:8000/stage-runs?project_id=car_single_smoke
curl -s http://localhost:8000/stage-runs/<stage_run_id>/events
```

The fake worker should move a COLMAP stage from `colmap_queued` to
`awaiting_colmap_approval`. Training stages move from `training_queued` to
`completed`.

### Compose

Start the whole local controller stack:

```bash
docker compose up --build
```

In another terminal, create a project and queue a fake stage:

```bash
curl -s -X POST http://localhost:8000/projects \
  -H 'content-type: application/json' \
  -d '{"id":"compose_smoke","name":"Compose smoke","raw_uri":"r2://buildvision3d-pipeline/projects/compose_smoke/raw"}'

curl -s -X POST http://localhost:8000/projects/compose_smoke/stage-runs \
  -H 'content-type: application/json' \
  -d '{"stage":"colmap","provider":"local_fake","input_uri_json":{"preprocess":"r2://buildvision3d-pipeline/projects/compose_smoke/preprocess/current"}}'
```

The always-running worker should claim it automatically. Inspect the result:

```bash
curl -s 'http://localhost:8000/stage-runs?project_id=compose_smoke'
```

Useful logs:

```bash
docker compose logs -f api
docker compose logs -f worker
```

## Workflow Actions

Approve a stage that is waiting at a gate:

```bash
curl -s -X POST http://localhost:8000/stage-runs/<stage_run_id>/approve \
  -H 'content-type: application/json' \
  -d '{"notes":"Looks good"}'
```

Approval transitions:

```text
awaiting_preprocess_approval -> colmap_queued
awaiting_colmap_approval -> training_queued
```

Reject, retry, or cancel a stage run:

```bash
curl -s -X POST http://localhost:8000/stage-runs/<stage_run_id>/reject \
  -H 'content-type: application/json' \
  -d '{"notes":"Need different keyframes"}'

curl -s -X POST http://localhost:8000/stage-runs/<stage_run_id>/retry \
  -H 'content-type: application/json' \
  -d '{"notes":"Try again"}'

curl -s -X POST http://localhost:8000/stage-runs/<stage_run_id>/cancel \
  -H 'content-type: application/json' \
  -d '{"notes":"Operator cancelled"}'
```

The fake provider now writes progress events at 25, 50, 75, and 100 percent.

## Postgres Scope

Postgres is intentionally only the controller state store. Keep it to:

```text
project state
stage queue/status
approvals and decisions
compact lifecycle/progress events
small summaries and artifact/log URIs
```

Do not store full stdout/stderr streams in Postgres. The `local_preprocess`
worker counts subprocess output lines and keeps a small final-line sample in
`summary_json`, but it does not insert per-line log events. Full logs should
live in stage artifacts or R2 once that log path is wired.

## Smoke Helper

For less typing during development:

```bash
python scripts/controller_smoke.py fake-chain --project-id smoke_8a
python -m controller_worker --once
python scripts/controller_smoke.py approve <preprocess_stage_run_id>
python -m controller_worker --once
python scripts/controller_smoke.py approve <colmap_stage_run_id>
python -m controller_worker --once
```

That proves:

```text
preprocess_queued -> awaiting_preprocess_approval
approve preprocess -> colmap_queued
colmap_queued -> awaiting_colmap_approval
approve colmap -> training_queued
training_queued -> completed
```

## Local Preprocess Integration

Real preprocessing uses R2 for durable inputs and outputs. The controller worker
may use temporary scratch space inside the stage wrapper, but `raw_uri` and
`output_uri` for `local_preprocess` must be `r2://` URIs.

Upload raw media through the API:

```bash
curl -s -X POST http://localhost:8000/projects/preprocess_smoke/raw \
  -F "name=Preprocess Smoke" \
  -F "files=@data/raw/preprocess_smoke/input.mp4"
```

To validate the upload request and manifest generation without writing to R2 or
updating project state:

```bash
curl -s -X POST http://localhost:8000/projects/preprocess_smoke/raw \
  -F "name=Preprocess Smoke" \
  -F "dry_run=true" \
  -F "files=@data/raw/preprocess_smoke/input.mp4"
```

Loose images can be marked as hero/detail images without using a `hero/`
folder by passing `metadata_json`:

```bash
curl -s -X POST http://localhost:8000/projects/preprocess_smoke/raw \
  -F "name=Preprocess Smoke" \
  -F 'metadata_json={"files":[{"filename":"detail_01.jpg","role":"hero_image","location":"kitchen","colmap_policy":"optional"}]}' \
  -F "files=@/path/to/detail_01.jpg;filename=detail_01.jpg"
```

Supported per-file metadata fields:

```text
filename or relative_path
source_id
role: coverage_video, coverage_image, hero_image
location
camera_group
colmap_policy: include, optional, exclude
related_sources
```

The API writes the uploaded files to temporary scratch space, builds
`sources_manifest.json` with the same manifest helper as
`scripts/upload_raw_project.py`, syncs everything to:

```text
r2://<bucket>/projects/preprocess_smoke/raw/
```

and creates or updates the controller project with `status=raw_uploaded`.

Queue real preprocessing through the API:

```bash
curl -s -X POST http://localhost:8000/projects/smoke_preprocess/preprocess \
  -H 'content-type: application/json' \
  -d '{
    "raw_uri": "r2://buildvision3d-pipeline/projects/smoke_preprocess/raw",
    "output_uri": "r2://buildvision3d-pipeline/projects/smoke_preprocess/preprocess",
    "profile": "indoor_room",
    "provider": "local_preprocess"
  }'
```

Or use the helper:

```bash
python scripts/controller_smoke.py queue-preprocess \
  --project-id smoke_preprocess \
  --raw-uri r2://buildvision3d-pipeline/projects/smoke_preprocess/raw \
  --output-uri r2://buildvision3d-pipeline/projects/smoke_preprocess/preprocess \
  --profile indoor_room
```

Then run:

```bash
python -m controller_worker --once
```

For a non-destructive command-shape test, add `--dry-run` to the helper. The
worker still exercises the command builder and event path, but the stage wrapper
does not run preprocessing or upload artifacts. Because dry-run does not upload
`preprocess_summary.json`, the worker skips the R2 summary fetch in that mode.

On success, the stage wrapper uploads:

```text
r2://<bucket>/projects/<project_id>/preprocess/current/frames_selected/
r2://<bucket>/projects/<project_id>/preprocess/current/capture_report.json
r2://<bucket>/projects/<project_id>/preprocess/current/image_manifest.json
r2://<bucket>/projects/<project_id>/preprocess/current/sources_manifest.json
r2://<bucket>/projects/<project_id>/preprocess/current/preprocess_summary.json
r2://<bucket>/projects/<project_id>/preprocess/current/stage_result.json
r2://<bucket>/projects/<project_id>/preprocess/runs/<stage_run_id>/capture_report.json
r2://<bucket>/projects/<project_id>/preprocess/runs/<stage_run_id>/preprocess_summary.json
r2://<bucket>/projects/<project_id>/preprocess/runs/<stage_run_id>/stage_result.json
```

After the wrapper exits, the worker reads only
`current/preprocess_summary.json` back from R2 and stores the compact summary in
`stage_runs.summary_json`.

## Compose Services

The Compose services use the same commands as terminal mode:

```yaml
api:
  build: .
  command: uvicorn controller_api.main:app --reload --host 0.0.0.0 --port 8000
  volumes:
    - .:/app
  env_file:
    - .env
  depends_on:
    - postgres

worker:
  build: .
  command: python -m controller_worker
  volumes:
    - .:/app
  env_file:
    - .env
  depends_on:
    - postgres
```

The repository is mounted into `/app` for live edits.
