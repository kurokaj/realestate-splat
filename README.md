# Buildvision3D

Script-first tooling for turning real estate capture videos or image sets into selected frames and reports for later Gaussian splatting.

## Milestone 1: local preprocessing

Put future source videos and optional root-level coverage images in a project folder. Keep high-detail hero photos under `hero/` so they stay separate from normal coverage images:

```text
data/raw/<splat_project_name>/
  kitchen.mp4
  living_room.mp4
  bedroom.mp4
  coverage_photo_001.jpg
  coverage_photo_002.jpg
  hero/
    kitchen/
      hero_001.jpg
```

Install the local dependencies with conda:

```bash
/opt/homebrew/bin/conda env create -f environment.yml
```

Or with a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run preprocessing:

```bash
python scripts/preprocess_video.py \
  --input-dir data/raw/apartment_001 \
  --profile small_apartment
```

The script writes:

```text
runs/apartment_001/
  frames_selected/
    kitchen_frame_000001.jpg
    living_room_frame_000001.jpg
    bedroom_frame_000001.jpg
  reports/
    capture_report.json
    capture_report.html
    gpu_recommendation.json
  run_config.json
```

It does not create a raw frame cache. Only final selected frames are written.
All selected frames from the project videos and selected root-level coverage
images go into one `frames_selected/` folder for a single downstream COLMAP and
splatfacto run. The HTML report summarizes each source video, the coverage
image quality checks, and the combined selected-image count. Use
`--out runs/custom_name` to override the inferred run directory. Selection
settings such as `--target-max` apply per source video and to the root coverage
image set.

For the current toy parking-garage capture, use:

```bash
/opt/homebrew/bin/conda run -n splat-dev-locat python scripts/preprocess_video.py \
  --input-dir data/raw/parkkihalli_dome_gap_aware \
  --out runs/parkkihalli_dome_gap_aware \
  --profile indoor_room \
  --candidate-fps 3.0 \
  --target-min 100 \
  --target-max 180 \
  --min-blur 40 \
  --force-keep-interval 1.0
```

Gap-aware selection is enabled by default with one frame per two-second window.
Coverage fallback frames are included in the report separately from normal quality-selected frames.
Tune this with:

```bash
--coverage-window-seconds 2.0
--min-frames-per-window 1
```

## Milestone 2: manual Verda pipeline

Upload the selected run directory to the Verda block volume, then SSH into the
instance and run COLMAP, splatfacto training, and export from the persistent
workspace:

`scripts/run_colmap.py` calls `/workspace/opt/colmap-install/bin/colmap` by
absolute path. Nerfstudio is only used through the Pixi checkout at
`/workspace/opt/nerfstudio` for Splatfacto training and export.

```bash
cd /workspace/repo/realestate-splat
source ~/.bashrc
micromamba activate /workspace/envs/splat-dev

python scripts/run_colmap.py \
  --run /workspace/runs/parkkihalli_dome_gap_aware \
  --config configs/training_splatfacto_dev.yaml

python scripts/run_training.py \
  --run /workspace/runs/parkkihalli_dome_gap_aware \
  --config configs/training_splatfacto_dev.yaml \
  --backend splatfacto \
  --max-steps 5000 \
  --num-downscales 2

python scripts/export_scene.py \
  --run /workspace/runs/parkkihalli_dome_gap_aware \
  --config configs/training_splatfacto_dev.yaml
```

The scripts write:

```text
runs/parkkihalli_dome_gap_aware/
  colmap/
    database.db
    database_global.db
    sparse/
    sparse_txt/
    logs/
  nerfstudio/
    transforms.json
    images/
  gsplat/
    outputs/
    exports/
    logs/
  reports/
    reconstruction_report.json
    training_report.json
    export_report.json
  final/
    scene.ply
    viewer_config.json
```

Use `--dry-run` locally to inspect commands without writing outputs.
For `global_mapper`, the COLMAP script copies `database.db` to
`database_global.db`, runs `view_graph_calibrator` on the copy, and maps from
the calibrated database.
PLY is the first canonical export. A viewer-specific `.splat` or SuperSplat /
PlayCanvas-compatible file should be added after the browser viewer choice is fixed.

## Milestone 3: local cloud orchestration - complete

Status: complete as of 2026-07-01. The pipeline has successfully run
preprocess, zip upload, remote COLMAP, remote gsplat training, export, and
artifact download through the CLI orchestrator.

Once a Verda host is reachable over SSH, run the scripted cloud pipeline from
the Mac:

```bash
python scripts/run_cloud_pipeline.py \
  --run runs/parkkihalli_dome_gap_aware \
  --host verda-a6000 \
  --approval-mode approve_warnings
```

Or preprocess first and then upload/run remotely:

```bash
python scripts/run_cloud_pipeline.py \
  --input-dir data/raw/apartment_001 \
  --out runs/apartment_001 \
  --profile small_apartment \
  --host verda-a6000
```

The pipeline zips `frames_selected/`, local capture reports, and `run_config.json`,
rsyncs that one bundle to Verda, unpacks it into `/workspace/runs/<run>/`, runs
remote COLMAP, training, and export over SSH, then zips final artifacts/reports/logs
remotely and rsyncs one return bundle into:

```text
runs/<run>/cloud_artifacts/
```

Transfer zips are scratch artifacts. After successful use, the pipeline removes
the local upload bundle, the remote upload bundle, the remote final-artifact
bundle, the downloaded final-artifact bundle, and the COLMAP review bundles
after they have been unpacked.

Use `--skip-upload` to resume after a successful upload/unpack when the remote
run inputs are already present.

Preflight policy is intentionally conservative:

- Hopeless captures fail locally before cloud spend starts.
- All non-fatal captures pause for operator approval after `reports/capture_report.html` is written.
- Warning-level captures show their review reasons in the prompt.
- After remote COLMAP finishes, `reconstruction_report.html/json` and `model_analyzer.log` are downloaded locally before training starts.
- The pipeline pauses again so the operator can review COLMAP metrics and decide whether to start training.
- Use `--yes-to-prompts` only for an intentionally non-interactive run.
- Use `--dry-run` to print the planned SSH/rsync commands without connecting.

Remote COLMAP now writes both `reports/reconstruction_report.json` and
`reports/reconstruction_report.html` so reconstruction quality can be reviewed
without reading raw logs first. The HTML/JSON include the parsed COLMAP
`model_analyzer` summary, such as registered images, point count, observation
count, mean track length, observations per image, and mean reprojection error.
