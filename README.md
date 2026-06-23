# Buildvision3D

Script-first tooling for turning real estate capture video into selected frames and reports for later Gaussian splatting.

## Milestone 1: local preprocessing

Put future source videos in:

```text
data/raw/
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
  --video data/raw/room_001.mp4 \
  --out runs/room_001 \
  --profile indoor_room
```

The script writes:

```text
runs/room_001/
  frames_selected/
  reports/
    capture_report.json
    capture_report.html
    frame_contact_sheet.jpg
    gpu_recommendation.json
  run_config.json
```

It does not create a raw frame cache. Only final selected frames are written.

For the current toy parking-garage capture, use:

```bash
/opt/homebrew/bin/conda run -n splat-dev-locat python scripts/preprocess_video.py \
  --video data/raw/parkkihalli.mp4 \
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
PLY is the first canonical export. A viewer-specific `.splat` or SuperSplat /
PlayCanvas-compatible file should be added after the browser viewer choice is fixed.
