# Controller Plan

## Purpose

The controller turns the proven stage scripts into a semi-automatic pipeline:

```text
upload raw media
  -> preprocess on local/CPU runtime
  -> approve selected frames
  -> start COLMAP GPU pod
  -> approve reconstruction
  -> start Nerfstudio GPU pod
  -> export splat
  -> inspect result in the app
```

The controller should stay small at first. It is a local/self-hosted tool for
running the owner's own projects, not a customer-facing SaaS backend.

## Current Building Blocks

Object storage:

```text
Cloudflare R2 bucket
projects/<project_id>/raw/
projects/<project_id>/preprocess/current/
projects/<project_id>/preprocess/runs/<stage_run_id>/
projects/<project_id>/colmap/current/
projects/<project_id>/colmap/runs/<stage_run_id>/
projects/<project_id>/training/current/
projects/<project_id>/training/runs/<stage_run_id>/
```

Stage scripts:

```text
scripts/upload_raw_project.py
scripts/run_preprocess_stage.py
scripts/run_colmap_stage.py
scripts/run_training_stage.py
scripts/sync_run_artifacts.py
```

Verified images:

```text
COLMAP:
docker.io/blackjokuro/buildvision3d-colmap-gpu:cuda12.4-colmap-r2-runtime-sm75-sm86-sm89-r2

Nerfstudio:
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:cuda11.8-pixi-splatfacto-r2-clean-sm75-sm86-sm89-r2
```

## Split Of Responsibilities

The controller owns orchestration:

```text
project state
stage state
provider choice
pod start/stop
environment variables/secrets injection
command construction
stdout/stderr tailing
progress/events storage
retry/cancel decisions
human approval gates
```

The stage scripts own execution:

```text
download needed inputs from R2
run the processing command
write stage_result.json
write stage summaries
upload current artifacts to R2
upload lightweight history metadata to R2
exit nonzero on failure
```

The GPU pod should be disposable. It does not own durable data. If the pod dies,
the controller should be able to show the failed stage and rerun it from R2.

## First Architecture

Start with one local stack: Postgres, a FastAPI backend, and a separate
worker-controller process. Postgres is preferred over SQLite because it matches
the future durable setup and makes state inspection easy.

```text
Local browser UI
        |
        v
FastAPI backend + controller loop
        |
        +--> Postgres state
        |
        +--> R2 artifacts
        |
        +--> local CPU preprocess command
        |
        +--> RunPod GPU pod: COLMAP image
        |
        +--> RunPod GPU pod: Nerfstudio image
```

Do not add Prefect or Temporal now. The pipeline is small enough that a direct
controller loop is easier to understand and cheaper to host.

## Development Runtime Path

Use a terminal-first shape before moving the same commands into full Compose.
This keeps debugging simple while avoiding code that only works on one local
machine path.

Phase 1:

```text
Postgres runs in Docker.
FastAPI runs from the terminal.
Worker/controller runs from the terminal.
```

Command shape:

```bash
docker compose up postgres
uvicorn controller_api.main:app --reload --host 0.0.0.0 --port 8000
python -m controller_worker
```

Phase 2:

```text
Postgres, FastAPI, and worker/controller all run under docker compose.
The API and worker use the same commands as Phase 1.
The repo is mounted into the API and worker containers for live Python edits.
```

Compose service shape:

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

The migration from Phase 1 to Phase 2 should be mostly moving process startup
into `docker-compose.yml`. Keep paths workspace-relative and put settings in
environment variables, especially `DATABASE_URL`, R2 credentials, and provider
tokens.

## Initial Data Model

Projects:

```text
id
name
created_at
updated_at
status
raw_uri
preprocess_current_uri
colmap_current_uri
training_current_uri
active_preprocess_run_id
active_colmap_run_id
active_training_run_id
```

Stage runs:

```text
id
project_id
stage
status
attempt
image
provider
provider_job_id
provider_pod_id
command
input_uri_json
output_uri
summary_json
progress_json
error_message
started_at
finished_at
created_at
updated_at
```

Events:

```text
id
stage_run_id
created_at
level
kind
message
payload_json
```

Approvals:

```text
id
project_id
stage
stage_run_id
status
decision
notes
created_at
decided_at
```

## Stage States

Use concrete states instead of vague `processing`:

```text
created
raw_uploaded
preprocess_queued
preprocess_running
awaiting_preprocess_approval
preprocess_rejected
colmap_queued
waiting_for_gpu_capacity
colmap_pod_starting
colmap_running
awaiting_colmap_approval
colmap_rejected
training_queued
training_pod_starting
training_running
exporting
completed
failed
cancelled
```

## First Controller Loop

Keep the first loop boring and observable:

```text
1. Find one queued stage.
2. Claim it with an atomic DB update.
3. Create a stage_run row.
4. Execute the local command or start a GPU pod.
5. Tail logs into events.
6. Poll provider status.
7. Read stage_result.json from R2 when the command finishes.
8. Mark the stage complete or failed.
9. Stop the GPU pod.
10. Wait for user approval before the next gated stage.
```

At first, run only one active GPU stage at a time. Parallel scheduling can wait.

## Provider Adapter Shape

The controller should call a small interface so RunPod is not hardwired into the
whole app:

```text
create_job(image, command, env, gpu_type, disk_gb) -> provider_job
get_job(provider_job_id) -> status
tail_logs(provider_job_id, cursor) -> log_lines
terminate_job(provider_job_id) -> result
```

First adapter:

```text
RunPod pod adapter
```

Later adapters:

```text
Vast.ai
Verda manual/SSH adapter if useful
local fake adapter for tests
```

## UI Shape

The first UI should be an operator console, not a polished customer product.

Project view:

```text
upload/raw media status
pipeline graph
current stage status
stage history
action buttons: run, approve, reject, retry, cancel
```

Stage report tabs:

```text
Preprocess:
  selected frames
  dropped frames
  blur/exposure/motion summaries
  hero image manifest/status

COLMAP:
  registered image count
  sparse model summary
  reconstruction report
  key warnings

Training:
  training summary
  export status
  splat.ply link
  latest checkpoint/output path
```

Terminal/log view:

```text
show live stdout/stderr while a stage is running
show logs only for failed runs after completion
show structured progress when available
```

## RunPod Command Picture

The controller should start a pod with the image and a command shaped like:

```bash
bash -lc '
  cd /workspace &&
  git clone --branch production-runtime-roadmap <repo-url> Buildvision3D &&
  cd Buildvision3D &&
  python3 scripts/run_colmap_stage.py ...'
```

For Nerfstudio:

```bash
bash -lc '
  cd /workspace &&
  git clone --branch production-runtime-roadmap <repo-url> Buildvision3D &&
  cd Buildvision3D &&
  python3 scripts/run_training_stage.py ...'
```

The current images already include the needed runtime tools:

```text
COLMAP image: awscli, git, Python, COLMAP, GLOMAP/CASPAR path
Nerfstudio image: awscli, git, Python, Pixi, Nerfstudio, gsplat, tiny-cuda-nn
```

## Environment Variables

Required for stage pods:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=auto
R2_ENDPOINT
R2_BUCKET
```

Provider credentials should live in local environment variables or a local
`.env` file that is never committed.

## Milestone 8A: Local Controller Skeleton

Goal: prove that DB state can drive the pipeline graph without starting real
GPU pods yet.

Deliverables:

```text
docker-compose.yml for Postgres
minimal FastAPI backend
project CRUD endpoints
stage_run CRUD/read endpoints
controller loop that claims fake queued stages
events table and event writer
local fake provider adapter
README with terminal-first startup commands
notes for moving the same API/worker commands into full docker compose
```

Success check:

```text
Create a project, enqueue fake COLMAP/training stages, watch states progress,
and see events update in the API.
```

## Milestone 8B: Local Preprocess Integration

Goal: run the real CPU preprocess stage from controller state.

Deliverables:

```text
project raw upload/import endpoint
preprocess queue button/API
controller handler for scripts/run_preprocess_stage.py
R2 output URI recorded on success
preprocess summary visible from API
approval/reject state transition
```

Success check:

```text
Upload or reference single_car raw media, run preprocess, approve selected
frames, and see preprocess/current in R2.
```

## Milestone 8C: RunPod COLMAP Adapter

Goal: have the controller start and stop a real COLMAP GPU pod.

Deliverables:

```text
RunPod provider adapter
COLMAP pod template config
GPU type allowlist and max price config
pod command builder for scripts/run_colmap_stage.py
log polling into events
R2 stage_result polling
terminate pod on success/failure
manual retry
```

Success check:

```text
From the UI/API, run COLMAP for car_single_smoke and see colmap/current update
in R2 without manually entering the pod.
```

## Milestone 8D: RunPod Training Adapter

Goal: have the controller start and stop a real Nerfstudio GPU pod.

Deliverables:

```text
Nerfstudio pod template config
pod command builder for scripts/run_training_stage.py
training progress/log event capture
exported splat.ply URI recorded
terminate pod on success/failure
manual retry
```

Success check:

```text
From the UI/API, run training for car_single_smoke and see
training/current/exports/splat.ply update in R2.
```

## Milestone 8E: First Operator UI

Goal: replace the old HTML reports with an app surface.

Deliverables:

```text
project list
project detail
pipeline graph
preprocess report
COLMAP report
training report
stage history
live logs for running stages
failed-run logs
approve/reject/retry/cancel buttons
```

Success check:

```text
The full smoke pipeline can be driven through the app, with manual approvals,
while the CLI scripts remain runnable for debugging.
```

## Open Design Questions

Decide these during implementation:

```text
Use local Postgres immediately or start with SQLite and migrate quickly?
Use RunPod Pods API directly first, or RunPod serverless/jobs if suitable?
How much stdout should be stored in DB events vs R2 logs?
Where should provider templates live: DB, YAML config, or both?
Should controller live inside the FastAPI process at first, or as a separate worker process?
```

Recommendation for the next task:

```text
Start with Postgres in docker compose, then run FastAPI and the separate worker-controller from terminal.
Keep API and worker commands Compose-ready from the first commit.
Use a YAML provider config first.
Store short structured events in Postgres.
Store full logs in R2 only for failed runs or active log tails.
```
