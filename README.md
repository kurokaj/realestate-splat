# Buildvision3D

Buildvision3D is an internal tool for turning uploaded videos, coverage images,
and hero images into reviewed COLMAP reconstructions and Gaussian splats.

## Current Pipeline

```text
upload raw sources to R2
  -> local CPU preprocessing
  -> approve every source location
  -> select single or hybrid matching strategy
  -> disposable RunPod COLMAP pod
  -> inspect reconstruction and camera groups
  -> approve COLMAP
  -> disposable RunPod Nerfstudio/Splatfacto pod
```

R2 is the durable store. Postgres stores controller state, compact summaries,
approvals, progress, and artifact pointers. Large media, logs, reconstructions,
and training outputs stay in R2 or the disposable stage pod.

## Start Locally

Create a virtual environment for command-line checks if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the local R2 credentials. Never commit
`.env`.

Start the controller stack:

```bash
docker compose up --build -d
```

Open [http://localhost:8000](http://localhost:8000).

The stack contains:

- `postgres`: controller state and compact events
- `api`: FastAPI API and server-rendered internal UI
- `worker`: queue worker, local preprocessing runner, and RunPod lifecycle manager

## Raw Upload

The UI accepts drag-and-drop uploads. The CLI is useful for large files or
retries:

```bash
python scripts/upload_raw_project.py \
  --project-id dev-smoke \
  --input-dir data/raw/dev-smoke \
  --destination-uri "r2://$R2_BUCKET/projects/dev-smoke/raw" \
  --endpoint-url "$R2_ENDPOINT"
```

Uploads are additive. The API updates `sources_manifest.json`, preserves source
metadata, and marks affected locations for preprocessing.

## Stage Contracts

The active stage wrappers are:

```text
scripts/upload_raw_project.py
scripts/run_preprocess_stage.py
scripts/run_colmap_stage.py
scripts/run_training_stage.py
scripts/preprocess_video.py
scripts/run_colmap.py
scripts/prepare_nerfstudio_from_colmap.py
```

Each real stage downloads only its approved inputs, writes a compact stage
summary, uploads durable artifacts to R2, publishes completion markers, and
returns a nonzero exit code on failure. Stage pods are disposable and must not
be treated as durable storage.

## Storage Layout

```text
projects/<project_id>/raw/
projects/<project_id>/preprocess/groups/<group>/current/
projects/<project_id>/preprocess/groups/<group>/runs/<stage_run_id>/
projects/<project_id>/colmap/current/
projects/<project_id>/colmap/runs/<stage_run_id>/
projects/<project_id>/training/current/
projects/<project_id>/training/runs/<stage_run_id>/
```

Approved preprocess groups are assembled inside the COLMAP and training pods;
the controller does not create a duplicate project-wide `preprocess/current`
tree for grouped projects.

## Documentation

- [Controller plan](docs/controller_plan.md)
- [Controller 8A operations](docs/controller_8a.md)
- [Production runtime architecture](docs/production_runtime_architecture.md)
- [Processing strategies](docs/processing_strategies.md)
- [Processing strategy implementation plan](docs/processing_strategy_plan.md)
- [Docker images](docs/docker_images.md)
- [RunPod GPU notes](docs/runpod_gpus.md)
- [Future LiDAR and publishing plan](docs/future_3dgs_aholo_lidar_plan.md)

## Verification

Run syntax checks without creating repository bytecode:

```bash
PYTHONPYCACHEPREFIX=/tmp/buildvision3d-pycache \
  python3 -m py_compile \
  controller_api/main.py controller_worker/main.py \
  controller_ui/routes.py controller_common/*.py \
  scripts/preprocess_video.py scripts/run_colmap.py \
  scripts/run_colmap_stage.py scripts/run_training_stage.py
```

Check the Compose services:

```bash
docker compose ps
docker compose logs --since=5m api worker
```
