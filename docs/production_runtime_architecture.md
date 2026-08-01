# Production Runtime Architecture

## Goal

Build a production-like but cost-conscious runtime for:

1. video/file upload
2. CPU preprocessing and keyframe selection
3. human approval
4. COLMAP GPU reconstruction
5. human/automatic quality gate
6. Gaussian Splatting / nerfstudio training
7. artifact export to object storage
8. result preview/download

The system should support multiple GPU providers, starting with:

* RunPod
* Vast.ai
* Verda

Avoid Azure for now because cost pressure is high.

---

## Current status

The legacy Verda pipeline has already proven the core processing path:

```text
local preprocessing
  -> capture report and human approval
  -> zipped upload to Verda over SSH
  -> COLMAP reconstruction and review report
  -> Nerfstudio / splatfacto training
  -> export and artifact download
```

That path remains valuable for capture-style testing. The new work should not
break it, but it should no longer make Verda's block volume the durability
model. Production durability moves to object storage, explicit stage manifests,
containerized execution, and provider adapters.

Legacy reference docs:

```text
docs/legacy/realestate_splat_project_plan.md
docs/legacy/verda_tool_status.md
```

Active production runtime work should happen in this document and in the new
runtime/container/orchestration code it describes.

---

## Core principle

This is not an API endpoint workload.

It is a batch workflow with long-running jobs, expensive GPU stages, artifacts, retries, and human approval.

Use:

```text
Object storage + Postgres state + worker/controller + provider adapters + UI
```

Do not make one giant long-running serverless endpoint.

---

## Recommended architecture

```text
User uploads video
        |
        v
Cloudflare R2
        |
        v
API / backend
        |
        v
Postgres project/job state
        |
        v
Worker/controller loop
        |
        +--> CPU preprocessing worker
        |
        +--> Human approval step
        |
        +--> GPU provider scheduler
                |
                +--> RunPod
                +--> Vast.ai
                +--> Verda
        |
        +--> COLMAP container
        |
        +--> Human / automated quality gate
        |
        +--> Gaussian Splatting container
        |
        v
Cloudflare R2 final artifacts
```

---

## Provider automation status

Implementation order should start with RunPod because it gives the best balance
of automation, container workflow, and developer experience for this phase.
Vast.ai can follow as a cheaper opportunistic pool. Verda stays useful as a
legacy/manual provider and can become an adapter later if automation proves
worth the effort.

### Vast.ai

Vast.ai is suitable for automated provisioning.

It supports API/CLI flows for searching offers, creating instances, and destroying instances. Instance creation is based on finding an available offer and accepting it; destroying an instance is also supported through API/CLI.

Use Vast.ai as a cheap marketplace backend.

Pros:

* usually very cheap
* many GPU types
* good fit for opportunistic batch jobs
* programmatic create/destroy possible

Cons:

* availability varies
* machine reliability varies
* provider quality is not uniform
* needs strict artifact upload/checkpointing
* do not rely on local disk persistence

Use case:

```text
Best cheap fallback / primary provider for long GPU jobs when availability exists.
```

---

### RunPod

RunPod is also suitable for automated provisioning.

RunPod supports creating, starting, stopping, listing, and deleting Pods through CLI, and also supports GraphQL API pod management.

Pros:

* better developer experience than Vast.ai
* easier container workflow
* good middle ground between cheap and reliable
* supports Pods and serverless options

Cons:

* usually more expensive than best Vast.ai offers
* community GPU availability can still vary
* serverless may not be ideal for multi-hour training

Use case:

```text
Primary production-like GPU provider if developer experience matters.
```

---

### Verda

Verda appears usable for automation, but this should be validated directly in your account.

Verda has CLI documentation describing scriptable infrastructure management, and the public CLI repository says it can manage VMs, volumes, SSH keys, startup scripts, and more. Verda also has a Terraform provider for managing Verda infrastructure.

Pros:

* Finnish / European provider
* currently cheap for your manual workflow
* likely good for long-running GPU jobs
* may offer persistent volumes

Cons:

* GPU availability may be limited
* automation maturity must be tested
* may not be as easy to abstract as RunPod/Vast.ai

Use case:

```text
Keep as one provider in the pool, but do not make the architecture Verda-specific.
```

---

## Storage choice

Use Cloudflare R2.

R2 is S3-compatible object storage and is attractive for this pipeline because
Cloudflare's pricing docs currently separate storage and operation charges from
direct R2 egress bandwidth, which is listed as free. Re-check pricing before
committing to customer-facing cost estimates:
https://developers.cloudflare.com/r2/pricing/

Why R2 fits:

* videos, images, COLMAP outputs, splats, logs, thumbnails
* provider-neutral
* accessible from Vast.ai, RunPod, Verda, local machine
* avoids cloud lock-in
* avoids expensive egress when users download/view results

Suggested buckets:

```text
r2://gs-pipeline-raw/
r2://gs-pipeline-preprocessed/
r2://gs-pipeline-colmap/
r2://gs-pipeline-training/
r2://gs-pipeline-results/
r2://gs-pipeline-logs/
```

Suggested artifact layout:

```text
/projects/{project_id}/raw/
  sources_manifest.json
/projects/{project_id}/preprocess/
  keyframes/
  keyframes.json
  image_manifest.json
  run_config.json
  preview_contact_sheet.jpg
  capture_report.json
/projects/{project_id}/colmap/
  sparse/
  database.db
  images/
  colmap_summary.json
  reconstruction_report.json
  reconstruction_report.html
/projects/{project_id}/train/
  checkpoints/
  config.yml
  metrics.jsonl
  training_report.json
/projects/{project_id}/results/
  model.splat
  model.ply
  preview.mp4
  viewer_manifest.json
/logs/{project_id}/{job_id}/
```

---

## Media and hero image manifests

Hero/detail images are already implemented in the legacy preprocessing path.
They are discovered under:

```text
data/raw/<project>/hero/<location>/
```

and represented in `reports/image_manifest.json` with fields such as `role`,
`source_id`, `location`, `source_path`, `camera_group`, dimensions, and quality
metrics. Keep this behavior, but make it less dependent on folder naming over
time.

Production should treat media manifests as the contract between capture input,
preprocessing, COLMAP, training, viewer, and future annotation tools.

Minimum production manifest responsibilities:

* identify raw video, coverage image, and hero/detail image sources
* keep stable source IDs independent of local filenames
* mark image role: `coverage`, `coverage_image`, `hero`, or future roles
* preserve location/room labels when known
* allow hero images to use different camera groups and intrinsics
* record whether each image should participate in COLMAP, viewer detail UI, or both
* support object-storage URIs as well as local development paths

The current folder convention remains a useful default importer. The durable
version should also accept an explicit manifest so hero images stay in the loop
even when filenames, folder names, or camera sources differ.

Example direction:

```json
{
  "project_id": "house_001",
  "sources": [
    {
      "source_id": "coverage_kitchen_walkthrough",
      "role": "coverage_video",
      "uri": "r2://gs-pipeline-raw/projects/house_001/kitchen.mp4",
      "location": "kitchen"
    },
    {
      "source_id": "hero_kitchen_detail_001",
      "role": "hero_image",
      "uri": "r2://gs-pipeline-raw/projects/house_001/hero/kitchen/hero_001.jpg",
      "location": "kitchen",
      "related_sources": ["coverage_kitchen_walkthrough"],
      "camera_group": "hero_kitchen_phone_main",
      "colmap_policy": "optional"
    }
  ]
}
```

---

## Orchestration

Use a simple Postgres-backed state machine and a self-built controller.

This is a private operator tool, not a customer-facing SaaS platform. The
orchestrator only needs to coordinate durable artifacts, disposable compute,
approval pauses, retries, and notifications. It does not need expensive managed
workflow infrastructure to start.

Decision:

```text
Use:
  FastAPI + Postgres + worker/controller

Do not use for now:
  Prefect
  Temporal
  Airflow
  managed workflow services
```

This keeps the system cheap to host locally or on a small VPS and keeps the
processing model visible in the application database.

Default MVP:

```text
FastAPI backend
  -> writes projects, runs, stage_runs, approvals, artifacts

Postgres
  -> durable source of workflow truth

worker/controller process
  -> polls or subscribes to runnable stage_runs
  -> launches CPU preprocessing locally/in a CPU container
  -> starts/stops RunPod GPU jobs
  -> uploads logs/results to R2
  -> moves jobs to approval/waiting/failed/completed states

UI
  -> upload, inspect reports, approve/reject, show status
```

The controller should be boring and explicit:

```text
find next runnable stage_run
claim it with a DB lock/status transition
execute the stage command or provider job
write stage_result.json and artifacts
update Postgres
send notification if human action is needed
repeat
```

Human-in-the-loop pauses are just states:

```text
AWAITING_KEYFRAME_APPROVAL
AWAITING_COLMAP_APPROVAL
WAITING_FOR_GPU_CAPACITY
```

### What Postgres owns

Postgres is not required for the first R2-backed stage scripts. In the current
phase, R2 JSON files are enough to prove the stage contract. Add Postgres after
COLMAP and training stage wrappers can run from object-storage inputs and write
stable `current/` and `runs/{stage_run_id}/` outputs.

Rule of thumb:

```text
R2 is the source of truth for artifacts and stage receipts.
Postgres is an index, queue, approval ledger, and control-plane cache.
```

Postgres should store small, queryable state:

```text
projects
pipeline_runs
stage_runs
approvals
artifacts
provider_jobs
notifications
cost estimates
```

Postgres should not store videos, frames, checkpoints, splats, or large logs.
Those live in R2. Postgres stores their URIs, status, metadata, and the current
workflow position.

### What the controller owns

The controller is a normal Python service or process. It can run locally, on a
home machine, or on a cheap VPS.

Responsibilities:

```text
claim runnable stage_runs
run CPU preprocessing commands/containers
start RunPod GPU jobs
watch provider job status
stream or fetch logs
parse progress events
upload stage_result.json and artifacts to R2
update Postgres state
send notifications
stop/terminate GPU resources
```

It should be safe to restart. On startup it should inspect Postgres and provider
state, then continue from explicit statuses instead of assuming in-memory state.

### Queue options

Start without a separate queue if possible:

```text
Postgres table as queue + one controller process
```

This is enough while one operator is running a few long jobs. Add a queue only
when the controller gets hard to reason about.

Reasonable next steps if needed:

* **Postgres queue only**: simplest and easiest to inspect.
* **Dramatiq/RQ/Celery**: useful when many short background tasks appear.

Avoid adopting a workflow engine until the simple controller clearly becomes
hard to maintain. The escape hatch is clean state transitions and stage
contracts, not committing early to an external orchestrator.

---

## Where orchestration should run

Options:

### Option A: Cheap VPS

Recommended first production-like setup.

Run on:

* Hetzner VPS
* Fly.io
* Render
* Railway
* small local server
* home lab machine

Services:

```text
FastAPI backend
Postgres
worker/controller
optional frontend
```

Pros:

* cheap
* provider-neutral
* simple
* no Azure cost
* easy to debug

Cons:

* you own more ops
* need backups
* need monitoring

---

### Option B: Local hosting

Good during development.

Run locally:

```text
docker compose up
```

Services:

```text
backend
postgres
ui
worker-controller
```

GPU jobs still run remotely.

This is a good architecture because the orchestrator does not need GPU.

---

### Option C: Managed workflow service

Not recommended for the current phase.

Managed workflow services can reduce operational work later, but the current
tool is personal, low-throughput, and cost-sensitive. Keep workflow state in
Postgres first and spend complexity budget on artifact contracts, containers,
RunPod automation, and the review UI.

---

## PostgreSQL provider

Start with one of these:

### Recommended: Neon

Good default choice.

Pros:

* cheap/free start
* serverless Postgres
* good developer experience
* easy backups/branching

Cons:

* serverless cold behavior can matter
* not ideal for extremely high-throughput workloads, which you do not have yet

---

### Alternative: Supabase

Use if you also want:

* auth
* admin UI
* simple storage
* realtime updates

Supabase may be attractive if you want to build the approval UI quickly.

---

### Self-hosted Postgres

Use if you choose a VPS and want the cheapest possible version.

Pros:

* cheapest
* simple for one-person project

Cons:

* backups are your responsibility

Recommended default:

```text
Neon for DB
Cloudflare R2 for artifacts
FastAPI + worker/controller on local machine or cheap VPS
```

---

## UI plan

Build a small web UI.

Recommended stack:

```text
Next.js or React + FastAPI backend
```

Alternative simpler stack:

```text
FastAPI + HTMX
```

The UI should show:

### Project list

```text
Project
Status
Created
Current stage
Provider
GPU type
Cost estimate
Last updated
```

### Project detail page

Sections:

```text
1. Raw upload
2. Preprocessing result
3. Keyframe preview
4. Approval buttons
5. COLMAP result
6. Training progress
7. Final splat preview/download
8. Logs
```

### Approval steps

Human approval points:

```text
After preprocessing:
- approve keyframes
- reject and rerun preprocessing
- upload manual keyframes

After COLMAP:
- approve reconstruction
- reject and change parameters
- retry with different GPU/provider
```

### Visual previews

Generate:

```text
keyframe contact sheet
COLMAP sparse reconstruction summary
training loss chart
training preview renders
final viewer link
```

For final splat viewing:

* host static viewer assets
* store splat in R2
* use signed URL or public object depending on privacy

Possible UI hosting:

```text
Cloudflare Pages
Vercel
Netlify
cheap VPS
```

For cost pressure, use:

```text
Cloudflare Pages + FastAPI on VPS
```

or

```text
single VPS running frontend + backend
```

---

## Provider abstraction

Create a common provider interface.

```python
class GpuProvider:
    def find_capacity(self, requirements: GpuRequirements) -> list[GpuOffer]:
        ...

    def create_instance(self, offer: GpuOffer, job_spec: JobSpec) -> Instance:
        ...

    def wait_until_ready(self, instance_id: str) -> InstanceStatus:
        ...

    def run_job(self, instance_id: str, command: str) -> JobRunResult:
        ...

    def stream_logs(self, instance_id: str) -> Iterator[str]:
        ...

    def terminate_instance(self, instance_id: str) -> None:
        ...

    def estimate_cost(self, offer: GpuOffer, expected_minutes: int) -> Money:
        ...
```

Implement:

```text
providers/vast.py
providers/runpod.py
providers/verda.py
providers/local.py
```

The rest of the app should never call Vast.ai, RunPod, or Verda directly.

It should call:

```text
GpuProviderScheduler.request_gpu_job(...)
```

---

## GPU provider scheduler

The scheduler should choose provider based on:

```text
required_vram_gb
gpu_family
max_price_per_hour
expected_duration_minutes
availability
reliability score
region
previous failures
```

Example requirements:

```json
{
  "stage": "training",
  "min_vram_gb": 24,
  "preferred_gpus": ["RTX 4090", "L40S", "RTX 3090", "A5000"],
  "max_price_per_hour": 1.50,
  "expected_minutes": 180,
  "allow_interruptible": true
}
```

Provider selection order example:

```text
1. Check RunPod secure/community
2. Check Vast.ai offers
3. Check Verda
4. If none available, mark job as WAITING_FOR_CAPACITY
5. Retry every N minutes
```

Do not fail the workflow just because one provider has no GPU.

---

## What happens when no GPU is available?

This is normal.

State should become:

```text
WAITING_FOR_GPU_CAPACITY
```

Then retry periodically.

Example policy:

```text
Retry every 5 minutes for 1 hour
Then every 15 minutes for 12 hours
Then notify user
```

UI should say:

```text
Waiting for GPU capacity matching:
- >=24GB VRAM
- max €1.50/hour
- provider: Vast.ai, RunPod, Verda
```

User can override:

```text
increase max price
allow weaker GPU
allow different provider
pause job
cancel job
```

This matters because Verda availability is often limited, and Vast.ai/RunPod marketplace capacity can also fluctuate.

---

## GPU sizing strategy

Do not use one large GPU type for every stage.

This does not mean buying hardware.

It means:

```text
Do not run every pipeline step on the same expensive GPU instance.
```

Instead:

```text
CPU preprocessing:
  CPU only

COLMAP:
  cheaper GPU, e.g. 12-24GB VRAM depending on scene

Gaussian Splat training:
  larger/faster GPU, e.g. 24GB+ VRAM
```

The scheduler should choose GPU per stage.

Example:

```text
COLMAP:
  preferred: RTX 3060, RTX 3090, RTX A4000, T4, L4
  max price: low

Training:
  preferred: RTX 4090, RTX 3090, L40S, A5000/A6000
  max price: higher
```

Reason:

```text
If COLMAP is only 2 minutes faster on a huge GPU, using that huge GPU is waste.
If training is 60–90 minutes faster on a huge GPU, it may be worth it.
```

Benchmark before locking this in.

---

## Container strategy

Do not make one huge image forever.

Start with one image if needed for speed, but move toward:

```text
preprocess-cpu
colmap-gpu
nerfstudio-gs-gpu
```

Each image should:

* read input from R2
* write output to R2
* write metrics/logs to R2
* exit cleanly
* not depend on persistent local disk
* support resume/checkpoint if possible

### COLMAP image build strategy

Use a dedicated CUDA image for reconstruction instead of installing COLMAP on a
mounted provider volume. The image should contain a pinned COLMAP build with:

* CUDA-enabled feature extraction and matching
* incremental mapper support
* global mapper support
* view graph calibration support for global reconstruction from video frames
* Ceres/SuiteSparse/MKL or equivalent solver dependencies selected deliberately
* a small runtime entrypoint that verifies the binary before running a project

Do not rely on distribution COLMAP packages for this stage. Linux packages are
often CPU-only or built with dependency choices that are hard to inspect. Build
from source in the image and record the COLMAP commit, CUDA version, CMake
options, and CUDA architectures in the image label and stage result metadata.

Standalone GLOMAP should not be treated as a separate long-term dependency
unless a specific benchmark proves it is needed. Upstream GLOMAP has been
migrated into COLMAP, where the relevant path is exposed through the global
mapper flow.

The first verified COLMAP image targets the current RunPod GPU pool rather than
legacy V100. Compile CUDA code for the GPU families we expect to rent:

```text
75  # T4 / Turing
86  # RTX 30xx / A10 / A4000 / A5000 / A6000 Ampere class
89  # RTX 40xx / Ada class
```

The verified image is:

```text
docker.io/blackjokuro/buildvision3d-colmap-gpu:cuda12.4-colmap-global-caspar-sm75-sm86-sm89-r1
sha256:41ff37f24dbce2064147436c739f7711b997e8c130f2a26d8a2d2b67db240e4f
```

The canonical Dockerfile is:

```text
docker/colmap-gpu/Dockerfile
```

If a V100-compatible image is needed later, build a separate image with
`CUDA_ARCHS=70;75;86;89` and `CASPAR_ENABLED=OFF`. Caspar requires architecture
75 or newer.

Provider scheduling should pick GPU types at runtime. Separate images are only
worth it if benchmarks show a meaningful cold-start, compatibility, or runtime
gain, or if a dependency forces a different CUDA major version.

The first image validation should run:

```bash
colmap -h
colmap feature_extractor -h
colmap exhaustive_matcher -h
colmap mapper -h
colmap global_mapper -h
colmap view_graph_calibrator -h
```

The image was manually verified on Verda RTX A6000 with `car_single` frames.
The next step is an R2-backed stage wrapper rather than more manual local
smoke-test work.

Recommended command shape:

```bash
python scripts/run_stage.py \
  --project-id PROJECT_ID \
  --stage train \
  --input-uri r2://... \
  --output-uri r2://... \
  --config-uri r2://...
```

---

## Stage contract

Each stage should behave like a disposable job:

```text
read declared inputs
  -> write stage outputs
  -> write reports, logs, metrics, and artifact manifest
  -> upload durable outputs
  -> exit with an explicit status
```

The same stage command should support:

* local development paths, such as `runs/<project>/`
* object-storage URIs, such as `r2://...`
* dry-run validation
* rerun/resume where the stage can safely support it

Stage commands should not assume a mounted Verda volume, a specific SSH host, or
a persistent provider disk. Provider local disk is scratch space.

Suggested stage entrypoints:

```text
preprocess:
  input: raw media manifest or input media URI
  output: selected frames, image_manifest.json, capture reports, run_config.json

colmap:
  input: selected frames, image_manifest.json, COLMAP config
  output: database, sparse model, reconstruction reports, model analyzer logs

train:
  input: selected frames, sparse model, training config, optional checkpoint
  output: checkpoints, metrics, training report, exported model inputs

export:
  input: trained model/checkpoint
  output: scene.ply, viewer artifacts, viewer_manifest.json, preview assets
```

Each stage should produce a machine-readable `stage_result.json` with:

```text
project_id
pipeline_run_id
stage_run_id
stage
status
started_at
finished_at
input_uris
output_uris
artifact_manifest_uri
logs_uri
metrics_uri
error_message
```

Initial Milestone 4 helpers live in:

```text
src/realestate_splat/storage.py
src/realestate_splat/stage_contract.py
src/realestate_splat/media_manifest.py
scripts/sync_run_artifacts.py
scripts/upload_raw_project.py
scripts/run_preprocess_stage.py
```

The first supported operation is mirroring an existing local run directory to
local/S3-compatible storage and fetching it back:

```bash
python scripts/sync_run_artifacts.py upload \
  --run runs/<project_id> \
  --destination-uri r2://gs-pipeline-preprocessed/projects/<project_id>/preprocess

python scripts/sync_run_artifacts.py download \
  --source-uri r2://gs-pipeline-preprocessed/projects/<project_id>/preprocess \
  --run runs/<project_id>_from_storage
```

The upload command writes `artifact_manifest.json` into the local run before
syncing. Use `--sha256` when content hashes are worth the extra time.

For `r2://` URIs, the helper uses the AWS CLI with `R2_ENDPOINT` as the
S3-compatible endpoint. Plain local paths are supported without AWS tooling.

For new production-style projects, upload raw capture media instead of old run
artifacts:

```bash
python3 scripts/upload_raw_project.py \
  --project-id <project_id> \
  --input-dir data/raw/<project_id> \
  --destination-uri r2://buildvision3d-pipeline/projects/<project_id>/raw
```

This command writes and uploads `sources_manifest.json` beside the raw media.
The manifest is the next pipeline contract consumed by CPU preprocessing.

The preprocessing stage command consumes that raw prefix and uploads app-ready
JSON outputs:

```bash
python3 scripts/run_preprocess_stage.py \
  --project-id <project_id> \
  --raw-uri r2://buildvision3d-pipeline/projects/<project_id>/raw \
  --output-uri r2://buildvision3d-pipeline/projects/<project_id>/preprocess \
  --profile indoor_room
```

Preprocess output layout:

```text
projects/<project_id>/preprocess/current/
  frames_selected/              # latest selected images for COLMAP/training
  capture_report.json           # latest full app report data
  image_manifest.json           # latest selected-image manifest for COLMAP
  sources_manifest.json         # original raw input manifest, when available
  preprocess_summary.json       # compact list/status summary
  stage_result.json             # latest preprocessing stage status

projects/<project_id>/preprocess/runs/<stage_run_id>/
  capture_report.json           # slim history for the app report view
  preprocess_summary.json       # app list/timeline summary for this attempt
  stage_result.json             # status receipt for completed/failed attempts
```

The legacy `capture_report.html` remains useful for the old local workflow, but
the production stage does not upload it. The app should render preprocessing
history from JSON. The latest selected frames and metadata live under
`current/`; older frame selections can be recreated by rerunning the stage with
the same inputs and settings if needed. Successful preprocessing logs are not
uploaded by default; failed runs write `logs/preprocess.log` and
`stage_result.json` under `current/` and keep `stage_result.json` in history.

---

## Immediate operator action points

Before container work starts, validate storage-backed artifacts on one known
good run:

1. Create Cloudflare R2 buckets or equivalent S3-compatible buckets for raw,
   preprocessed, COLMAP, training, results, and logs.
2. Configure AWS-style credentials locally and set `R2_ENDPOINT`.
3. Run `scripts/sync_run_artifacts.py upload --dry-run` against one existing
   `runs/<project_id>/` directory.
4. Run the real upload for that same run.
5. Download it into a fresh local run directory and compare the expected
   reports, `frames_selected/`, `reports/image_manifest.json`, and `final/`
   artifacts.
6. Decide whether each stage should use separate buckets or one bucket with
   stage prefixes before the container images are built.

---

## Checkpointing

Training must checkpoint.

Minimum:

```text
Upload checkpoint every 10–15 minutes
Upload metrics every 1–5 minutes
Upload logs continuously or at job end
```

If a provider kills the instance or job fails:

```text
new GPU instance
download latest checkpoint
resume training
```

Without checkpointing, cheap marketplace GPUs become painful.

---

## State model

Postgres tables:

```sql
projects
- id
- name
- status
- created_at
- updated_at
- raw_input_uri
- current_stage
- user_id

pipeline_runs
- id
- project_id
- status
- started_at
- finished_at
- error_message

stage_runs
- id
- pipeline_run_id
- stage
- status
- provider
- provider_job_id
- gpu_name
- price_per_hour
- progress_json
- input_uris_json
- output_uris_json
- logs_uri
- events_uri
- stage_result_uri
- error_message
- claimed_by
- claimed_at
- started_at
- finished_at
- retry_count

approvals
- id
- project_id
- stage
- status
- approved_by
- created_at
- decided_at
- notes

artifacts
- id
- project_id
- stage_run_id
- stage
- type
- uri
- metadata_json
- created_at

provider_jobs
- id
- stage_run_id
- provider
- provider_job_id
- provider_instance_id
- status
- gpu_name
- price_per_hour
- region
- created_at
- started_at
- finished_at
- raw_status_json

notifications
- id
- project_id
- stage_run_id
- type
- status
- target
- message
- created_at
- sent_at

gpu_offers_cache
- id
- provider
- gpu_name
- vram_gb
- price_per_hour
- region
- available
- observed_at
```

---

## Pipeline states

Use explicit states.

```text
CREATED
UPLOADING
UPLOADED
PREPROCESSING
AWAITING_KEYFRAME_APPROVAL
PREPROCESS_REJECTED
COLMAP_QUEUED
WAITING_FOR_GPU_CAPACITY
COLMAP_RUNNING
AWAITING_COLMAP_APPROVAL
COLMAP_REJECTED
TRAINING_QUEUED
TRAINING_RUNNING
EXPORTING
COMPLETED
FAILED
CANCELLED
```

Do not use vague states like `processing`.

---

## Controller implementation outline

Start with one controller process. Do not build distributed scheduling until one
process is not enough.

Core loop:

```text
while true:
  find one runnable stage_run
  claim it with an atomic Postgres update
  execute stage handler
  write logs/events/stage_result
  update stage_run and project state
  sleep briefly
```

Runnable stages:

```text
UPLOADED -> PREPROCESSING
APPROVED_KEYFRAMES -> COLMAP_QUEUED
COLMAP_QUEUED -> WAITING_FOR_GPU_CAPACITY or COLMAP_RUNNING
APPROVED_COLMAP -> TRAINING_QUEUED
TRAINING_QUEUED -> WAITING_FOR_GPU_CAPACITY or TRAINING_RUNNING
TRAINING_COMPLETE -> EXPORTING
```

Claiming rule:

```sql
UPDATE stage_runs
SET status = 'RUNNING',
    claimed_by = :controller_id,
    claimed_at = now()
WHERE id = :stage_run_id
  AND status IN ('QUEUED', 'RETRY_READY')
RETURNING *;
```

For provider jobs, `RUNNING` means the controller owns the orchestration, not
that the GPU container is always actively computing. The provider job ID and
pod/instance metadata should be recorded so the controller can reconnect after
a restart.

Progress and logs:

```text
stdout/stderr -> logs/<project_id>/<stage_run_id>/stdout.log
structured events -> logs/<project_id>/<stage_run_id>/events.jsonl
latest progress summary -> stage_runs.progress_json
```

The UI can show the pipeline graph from Postgres state, terminal output from
log tails, and progress bars from `progress_json` or parsed `events.jsonl`.

Keep the first controller deliberately small:

```text
one process
one active local CPU stage at a time
one active GPU provider job per project
manual retry button before clever retry policy
manual cancel button before elaborate cancellation tree
```

---

## Cost tracking

Track estimated and actual costs.

For each GPU run:

```text
provider
gpu_name
price_per_hour
start_time
end_time
estimated_cost
actual_cost_estimate
```

At project level:

```text
preprocessing cost
COLMAP cost
training cost
storage cost estimate
total estimate
```

UI should show:

```text
This reconstruction cost approximately €X.YY in GPU time.
```

This helps you optimize based on cost per completed reconstruction.

---

## Local development

Use Docker Compose:

```text
services:
  postgres
  backend
  frontend
  worker-controller
```

Local `.env`:

```text
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=
R2_ENDPOINT=

VAST_API_KEY=
RUNPOD_API_KEY=
VERDA_API_KEY=

DATABASE_URL=
```

Add provider flags:

```text
ENABLE_VAST=true
ENABLE_RUNPOD=true
ENABLE_VERDA=false
ENABLE_LOCAL=true
```

---

## Development milestones

The old Verda-oriented plan reached Milestone 3. The viewer and segmentation
work from that plan are still product goals, but the active next milestone is
the runtime transition. Keep the working Verda path available while building
the new path beside it.

### Milestone 4: Storage-Backed Stage Contracts

Goal: stop treating local disk or a Verda mounted volume as the durable system.

Deliverables:

```text
object storage URI helpers
canonical R2/S3 artifact layout
sources/media manifest contract
stage_result.json contract
local-path and r2:// path support where practical
upload/download commands for run artifacts
documentation for mapping existing runs/ layout to object storage
```

Success criteria:

```text
An existing local run can be mirrored to object storage.
Preprocess, COLMAP, training, and export artifacts have stable URIs.
Hero images remain represented in image_manifest.json.
The legacy Verda pipeline still works from local paths.
```

### Milestone 5: Containerized CPU Preprocessing

Goal: prove one disposable stage using the cheapest runtime.

Deliverables:

```text
docker/preprocess-cpu image
stage command for preprocessing from local path or object storage
selected frames upload
capture_report.json upload
image_manifest.json upload
run_config.json upload
preprocess_summary.json upload
stage_result.json upload
local Docker run instructions
```

Success criteria:

```text
Given raw media in object storage, the container writes selected frames and reports back to object storage.
The same project can still be preprocessed locally for fast debugging.
```

### Milestone 6: Containerized COLMAP GPU Stage

Goal: isolate reconstruction from the training environment.

Deliverables:

```text
[done] docker/colmap-gpu image
[done] CUDA COLMAP with required mapper/global_mapper support
[done] incremental/global mapper support
[done] Ceres CUDA/cuDSS linkage
[done] Caspar-enabled RunPod-target image
[done] stage command that pulls selected frames and image_manifest.json
[done] manifest camera-group support in the stage wrapper
[done] reconstruction_report.json upload
[done] compact current/history COLMAP stage outputs
[pending] manual RunPod run documentation
```

Success criteria:

```text
COLMAP image runs manually on GPU and reconstructs the car_single smoke scene. [done]
COLMAP runs manually on a RunPod GPU from object-storage inputs. [pending]
Outputs can be uploaded without relying on provider-local persistent disk. [done in wrapper, pending GPU smoke]
Hero image registration/dropped counts remain visible in the reconstruction report. [done in wrapper, pending GPU smoke]
```

### Milestone 7: Containerized Nerfstudio Training Stage

Goal: reduce cold start and separate splat training from COLMAP.

Deliverables:

```text
[done] docker/nerfstudio-splatfacto-gpu image
[done] baked Nerfstudio/Pixi runtime
[done] torch CUDA, gsplat, tiny-cuda-nn verification
[done] short splatfacto smoke training
[done] stage command that pulls Nerfstudio/COLMAP inputs from object storage
[done] manual R2 smoke from preprocess/current + colmap/current
[done] manual R2 smoke exported training/current/exports/splat.ply
[pending] checkpoint upload every 10-15 minutes
[pending] metrics/log upload during or after training
[pending] resume from latest checkpoint
[done] export to canonical current PLY artifact
[pending] promotion of final deliverables/versioned exports
[pending] manual RunPod run documentation
[pending] optional Nerfstudio prepare mode for physical image undistortion
```

Initial implementation plan:

```text
1. Build Pixi-style image. [done]
2. Verify torch.cuda, gsplat, tiny-cuda-nn, ns-process-data, and ns-train. [done]
3. Run a short training smoke test from existing COLMAP text output. [done]
4. Add stage wrapper for object-storage inputs/outputs. [done]
5. Add checkpoint upload/resume behavior before provider automation. [pending]
```

Success criteria:

```text
Training image runs manually on GPU and completes a short splatfacto smoke. [done]
Training runs manually on RunPod from object-storage COLMAP outputs. [done]
A failed/interrupted job can resume from the latest uploaded checkpoint. [pending]
Cold start is materially lower than rebuilding the current Verda-style environment. [done]
```

Prepare-mode TODO:

```text
Current mode: transforms_only
  copies registered original images
  writes COLMAP intrinsics and distortion parameters into transforms.json
  relies on Nerfstudio's camera/distortion handling during loading/training

Future mode: undistort
  physically undistorts registered images
  updates intrinsics/transforms for the undistorted image space
  compares quality against transforms_only on one real capture
```

### Milestone 8: RunPod CLI Orchestrator

Goal: automate the GPU lifecycle without committing to the full UI/backend yet.

Deliverables:

```text
RunPod provider adapter
capacity search for target GPU/VRAM/price
create/start/wait/terminate pod flow
run COLMAP container job
run training container job
log/status fetch
cost estimate and actual runtime tracking
WAITING_FOR_GPU_CAPACITY handling
CLI approval gates after preprocessing and COLMAP
```

Success criteria:

```text
A CLI command can run preprocessing, wait for approval, start a RunPod COLMAP job, wait for approval, start a RunPod training job, export artifacts, and terminate GPU resources.
If capacity is unavailable, the job waits/retries instead of failing immediately.
```

Current implementation order:

```text
1. R2-backed COLMAP stage script. [done]
2. R2-backed training stage script.
3. Thin local controller CLI that reads stage_result.json and summary JSON from R2.
4. RunPod pod/job automation.
5. Postgres/Supabase only after the file-based stage contract is stable.
6. Local UI once the controller has stable status and history data to display.
```

### Milestone 9: Minimal Review UI

Goal: move human-in-the-loop review out of terminal prompts.

Deliverables:

```text
project list
project detail page
raw upload or import-from-storage flow
capture report preview and approval
COLMAP report preview and approval
training progress/log view
final artifact links
basic cost/status display
```

Success criteria:

```text
An operator can approve preprocessing and COLMAP from the UI while the same stage commands remain CLI-runnable.
```

### Milestone 10: Provider Pool and Durable Controller

Goal: move from one provider adapter to a resilient production workflow.

Deliverables:

```text
provider scheduler
Vast.ai adapter
Verda adapter if automation proves worthwhile
provider priority and max-price config
fallback behavior
Postgres job/project state
DB-backed retry/timeouts/cancel handling
controller restart recovery
```

Success criteria:

```text
If RunPod has no suitable GPU, the workflow can wait, try another provider, or expose a clear operator override.
Workflow state survives backend/controller restart.
```

### Deferred product milestones

These remain important, but should not block the runtime transition:

```text
browser viewer prototype
distance measurement
segmentation prototype
semantic labels / hotspots
full polished customer UI
```

---

## Important design rules

1. Never depend on provider local disk for final data.
2. Always upload artifacts to R2.
3. Always terminate GPU instances after job completion or failure.
4. Always checkpoint long training.
5. Always separate orchestration from execution.
6. Always allow provider fallback.
7. Always make stages rerunnable.
8. Do not optimize for serverless first.
9. Optimize for cost per completed reconstruction.
10. Keep Azure out unless customer or enterprise requirements force it.

---

## First implementation task

Build the foundation for Milestone 4 without disturbing the legacy Verda flow:

```text
Create storage-backed stage contracts and artifact helpers.

Include:
- docs for the canonical object-storage artifact layout
- a small storage abstraction that can support local paths and S3/R2 URIs
- commands to mirror an existing runs/<project>/ directory to object storage
- commands to fetch stage artifacts back to a local run directory
- stage_result.json schema/helpers
- manifest handling notes for raw media, selected frames, and hero images

Do not implement the full UI yet.
Do not implement all provider APIs yet.
Do not remove the existing Verda SSH pipeline.
Focus on making artifacts durable and stages rerunnable.
```

Next implementation task after R2 smoke testing:

```text
Create the self-hosted orchestration skeleton.

Include:
- docker-compose.yml with Postgres, backend, worker-controller
- initial SQL schema or migration files
- FastAPI project/status endpoints
- controller loop that can claim and complete fake stage_runs
- progress_json and log URI fields on stage_runs
- local-only fake provider for dry-run development
- README instructions for running everything locally

Do not integrate RunPod yet.
Do not build a polished UI yet.
Do not add Prefect/Temporal.
Prove that Postgres state can drive the pipeline graph first.
```

---

## Final recommendation

Use this stack first:

```text
Storage:
  Cloudflare R2

Database:
  Neon Postgres, or self-hosted Postgres on cheap VPS

Backend:
  FastAPI

Frontend:
  Next.js or FastAPI + HTMX

Orchestration:
  Postgres-backed state machine and worker/controller first
  external workflow engines are deferred

GPU providers:
  RunPod
  Vast.ai
  Verda

Hosting:
  cheap VPS or local machine first
```

The architecture should treat GPU providers as disposable capacity pools.

The product should own:

```text
state
artifacts
workflow
approvals
cost tracking
```

The GPU provider should only provide temporary compute.
