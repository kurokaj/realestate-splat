# Dynamic GPU Provider Pipeline Plan

## Goal

Build a production-like but cost-conscious pipeline for:

1. video/file upload
2. CPU preprocessing and keyframe selection
3. human approval
4. COLMAP GPU reconstruction
5. human/automatic quality gate
6. Gaussian Splatting / nerfstudio training
7. artifact export to object storage
8. result preview/download

The system should support multiple GPU providers, starting with:

* Vast.ai
* RunPod
* Verda

Avoid Azure for now because cost pressure is high.

---

## Core principle

This is not an API endpoint workload.

It is a batch workflow with long-running jobs, expensive GPU stages, artifacts, retries, and human approval.

Use:

```text
Object storage + workflow orchestrator + provider adapters + workers + UI
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
Postgres job state
        |
        v
Workflow orchestrator
        |
        +--> CPU preprocessing worker
        |
        +--> Human approval step
        |
        +--> GPU provider scheduler
                |
                +--> Vast.ai
                +--> RunPod
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

R2 is S3-compatible object storage and is especially attractive because it has zero egress fees. Cloudflare’s own pricing docs describe R2 pricing separately for storage and operations, and Cloudflare markets R2 around zero egress.

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
/projects/{project_id}/preprocess/
  keyframes/
  keyframes.json
  preview_contact_sheet.jpg
/projects/{project_id}/colmap/
  sparse/
  database.db
  images/
  colmap_summary.json
/projects/{project_id}/train/
  checkpoints/
  config.yml
  metrics.jsonl
/projects/{project_id}/results/
  model.splat
  model.ply
  preview.mp4
  viewer_manifest.json
/logs/{project_id}/{job_id}/
```

---

## Orchestration

### Recommended: Temporal

Use Temporal if you want the most robust version.

Temporal should not run GPU work directly. It should coordinate state.

Temporal workflow example:

```text
CreateProjectWorkflow
  -> wait for upload complete
  -> run PreprocessActivity
  -> wait for human approval
  -> run ColmapActivity
  -> evaluate COLMAP result
  -> wait for approval or auto-continue
  -> run TrainingActivity
  -> run ExportActivity
  -> mark project complete
```

Temporal is good because it supports:

* long-running workflows
* retries
* timers
* human-in-the-loop pauses
* durable workflow state
* resumability after crashes
* clean separation between orchestration and execution

Temporal should store workflow state. Postgres should store app/job/project state.

---

### Alternative: Prefect

Use Prefect if you want something easier to start with.

Pros:

* simpler Python-first orchestration
* nice UI
* easier mental model
* good for data-pipeline style workflows

Cons:

* human approval and long-running state are not as clean as Temporal
* less ideal for product workflows with many pauses/retries

Use Prefect if you want speed of implementation.

Use Temporal if you want the architecture you can grow into.

---

### Simple MVP alternative

For the first version, you can avoid Temporal and use:

```text
FastAPI + Postgres + background workers + queue
```

Possible queue choices:

* Redis Queue
* Dramatiq
* Celery
* simple Postgres queue

This is acceptable for MVP.

But design the state machine so Temporal can replace it later.

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
Temporal server or Prefect server
Postgres
Worker controller
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
temporal
ui
worker-controller
```

GPU jobs still run remotely.

This is a good architecture because the orchestrator does not need GPU.

---

### Option C: Managed Temporal Cloud

Best later, not first.

Pros:

* less orchestration ops
* reliable

Cons:

* additional cost
* maybe unnecessary early

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
Temporal or Prefect on VPS/local
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

Recommended command shape:

```bash
python run_stage.py \
  --project-id PROJECT_ID \
  --stage train \
  --input-uri r2://... \
  --output-uri r2://... \
  --config-uri r2://...
```

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
- provider_instance_id
- gpu_name
- price_per_hour
- started_at
- finished_at
- logs_uri
- output_uri
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
- stage
- type
- uri
- metadata_json
- created_at

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
  temporal
  backend
  frontend
  worker-cpu
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
TEMPORAL_ADDRESS=
```

Add provider flags:

```text
ENABLE_VAST=true
ENABLE_RUNPOD=true
ENABLE_VERDA=false
ENABLE_LOCAL=true
```

---

## MVP build order

### Phase 1: Make current manual pipeline reproducible

Deliverables:

```text
Docker image for current pipeline
R2 upload/download helpers
one CLI command that runs the full pipeline
config file per project
```

Success criterion:

```text
Given input video in R2, run pipeline manually on one GPU VM and write outputs back to R2.
```

---

### Phase 2: Split stages

Deliverables:

```text
preprocess command
colmap command
train command
export command
```

Success criterion:

```text
Each stage can be rerun independently from R2 artifacts.
```

---

### Phase 3: Add database and API

Deliverables:

```text
FastAPI backend
Postgres schema
project creation
artifact registration
status updates
```

Success criterion:

```text
UI/API can show project status and artifacts.
```

---

### Phase 4: Add UI

Deliverables:

```text
project list
project detail page
keyframe preview
approve/reject buttons
logs view
final result links
```

Success criterion:

```text
Human can approve preprocessing before GPU work starts.
```

---

### Phase 5: Add one provider adapter

Start with RunPod or Vast.ai.

Recommendation:

```text
RunPod first if you want easier implementation.
Vast.ai first if you want cheapest-first behavior.
```

Deliverables:

```text
find offers
create instance/pod
wait ready
run command
upload logs
terminate
```

Success criterion:

```text
COLMAP or training can run from the orchestrator without manual VM setup.
```

---

### Phase 6: Add scheduler

Deliverables:

```text
provider priority config
capacity search
max price rules
fallback provider logic
WAITING_FOR_GPU_CAPACITY state
```

Success criterion:

```text
If provider A has no GPU, provider B is tried automatically.
```

---

### Phase 7: Add Temporal

Deliverables:

```text
Temporal workflow
activities for each stage
human approval wait states
retry policies
timeouts
```

Success criterion:

```text
Workflow survives backend restart and resumes correctly.
```

---

### Phase 8: Add Verda adapter

Deliverables:

```text
Verda CLI/API integration
create VM/job
attach or configure storage
run startup script
terminate resource
```

Success criterion:

```text
Verda can be used as one selectable provider, not a special-case manual path.
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

## First coding-agent task

Ask the coding agent to implement this first:

```text
Create a repository skeleton for the GPU reconstruction pipeline.

Include:
- FastAPI backend
- Postgres schema
- R2 storage client
- provider abstraction interface
- placeholder providers for Vast.ai, RunPod, Verda, Local
- pipeline state machine
- CLI commands for preprocess, colmap, train
- Docker Compose for local development
- README with setup instructions

Do not implement the full UI yet.
Do not implement all provider APIs yet.
Focus on clean interfaces and state transitions.
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
  Temporal eventually
  Simple queue or Prefect for MVP if needed

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
