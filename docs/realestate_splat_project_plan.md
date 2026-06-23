# Real Estate Gaussian Splatting Pipeline Plan

## 1. Product Goal

Build a repeatable pipeline for creating browser-hosted Gaussian splatting scenes for real estate walkthroughs.

The target experience is a “gamified” property visit:

- user opens a shared link
- navigates the space with WASD + mouse
- explores the home like a simple 3D world
- optionally uses lightweight AR-like features, such as:
  - rough distance measurements between two clicked points
  - object/room annotations
  - segmented areas or objects
  - guided hotspots or labels

The goal is not millimeter-accurate surveying. The goal is a visually convincing, navigable, immersive property walkthrough.

Initial capture process is controlled manually by the developer/operator. This is important because the pipeline can assume reasonable capture quality and does not need to handle arbitrary bad user uploads in the first version.

---

## 2. High-Level Architecture

The system should be built in layers.

```text
Local machine
  ↓
Video preprocessing + frame selection
  ↓
Capture report + GPU recommendation
  ↓
Upload selected frames to cloud GPU
  ↓
COLMAP reconstruction
  ↓
Gaussian splat training
  ↓
Export scene artifact
  ↓
Upload to hosting/storage
  ↓
Browser viewer
```

Do not start with a full UI app. Start with reliable scripts. Add UI later.

---

## 3. Development Philosophy

The pipeline should be script-first, not notebook-first.

Use scripts for:

- preprocessing
- frame selection
- COLMAP execution
- gsplat training
- artifact export
- upload/download
- automation

Use notebooks only for:

- inspecting frame quality
- plotting metrics
- debugging selected frames
- visualizing sparse point clouds
- comparing reconstruction settings
- prototyping segmentation projection

The goal is that the entire pipeline can eventually run from one command.

---

## 4. Main Pipeline Stages

### Stage 1: Local Video Preprocessing

Runs locally on the Mac.

Input:

```text
raw video file
```

Output:

```text
selected frames
capture report
GPU recommendation
run config
```

Responsibilities:

- extract candidate frames from video
- remove blurry frames
- remove too dark / too bright frames
- remove near-duplicate frames
- select enough frames for COLMAP but not too many
- create visual report
- estimate GPU tier needed

This stage should be lightweight and local because it does not require a large GPU.

---

### Stage 2: Frame Selection Logic

Frame selection should initially be rule-based.

Metrics to compute per frame:

- blur score
- brightness
- contrast
- entropy / texture score
- timestamp
- perceptual hash or embedding similarity to previous selected frame

Initial selection rules:

```text
drop very blurry frames
drop very dark / overexposed frames
drop near-duplicates
keep enough temporal coverage
use gap-aware fallback frames when a time window would otherwise be empty
keep frames around turns and room transitions
avoid selecting every video frame
```

Gap-aware frame selection is important for reconstruction. The local
preprocessing script should prefer high-quality frames, but it should not
silently skip an entire portion of the camera path just because every frame in
that portion narrowly missed a normal threshold. The first implementation uses
2-second coverage windows and keeps at least one frame per window when possible.
Fallback frames must still pass hard safety thresholds so truly unusable frames
remain rejected. Reports should clearly distinguish normal quality-selected
frames from coverage fallback frames.

Suggested extraction / selection targets:

| Scene Type | Suggested Selected Frames |
|---|---:|
| Single room | 100–250 |
| Small apartment | 300–700 |
| 50–120 m² house | 800–1,800 |
| Outdoor drone orbit | 300–900 |
| Indoor + outdoor combined | split first; 1,500–3,000 later |

For early development, avoid giant full-house reconstructions. Start with room-level and floor-level tests.

---

### Stage 3: Capture Report

Every preprocessing run should produce a report.

Output files:

```text
reports/capture_report.json
reports/capture_report.html
reports/frame_contact_sheet.jpg
reports/gpu_recommendation.json
```

The report should include:

- original video duration
- extracted candidate frame count
- selected frame count
- selected-by counts, such as quality vs coverage fallback
- rejected blurry frame count
- rejected duplicate frame count
- blur score distribution
- brightness distribution
- selected frame timeline
- coverage window summary
- largest selected-frame timeline gap
- estimated scene size
- recommended GPU tier
- warning flags

Example warning flags:

```text
too few selected frames
too many selected frames
high blur rate
coverage gaps remain
target max below coverage minimum
low texture scene
lighting unstable
possible fast camera motion
possible fisheye / wide distortion
```

---

### Stage 4: GPU Recommendation

The local preprocessing stage should estimate what GPU to use.

Initial heuristic:

| Selected Images | Resolution | Suggested GPU |
|---:|---|---|
| <300 | 1080p–2K | 24–40 GB VRAM |
| 300–800 | 1080p–2K | 40–48 GB VRAM |
| 800–1,500 | 1080p–2K | 48–80 GB VRAM |
| 1,500–2,500 | 2K+ | 80–96 GB VRAM |
| 2,500+ | high-res / large scene | 96 GB+ or split scene |

The recommender should be conservative. It is better to pay slightly more once than to waste GPU time debugging out-of-memory failures.

Recommended Verda strategy:

```text
development / tests:
  RTX A6000 48GB or similar

real full-house runs:
  RTX PRO 6000 96GB or A100 80GB

avoid initially:
  H100 / H200 / B-series unless evidence shows 96GB is not enough
```

---

### Stage 5: Cloud GPU Reconstruction

Runs on Verda GPU instance.

Inputs:

```text
frames_selected/
colmap config
training config
```

Outputs:

```text
COLMAP database
sparse reconstruction
camera parameters
point cloud
logs
```

COLMAP should be executed through the CLI, not only through Python bindings.
The authoritative COLMAP binary on Verda is:

```text
/workspace/opt/colmap-install/bin/colmap
```

Pipeline scripts should call this binary by absolute path. Nerfstudio's Pixi
environment may contain its own COLMAP dependency, but that dependency must not
own the reconstruction stage.

Primary reconstruction modes:

```text
incremental mapper
global_mapper
```

The pipeline should support both:

```yaml
colmap:
  mode: global  # incremental | global
```

For real estate scenes, start with incremental mapping as baseline, then test global mapping. Global mapping may be useful for larger scenes but should not be assumed superior in every capture.

---

### Stage 6: Gaussian Splat Training

Runs on Verda GPU instance.

Inputs:

```text
selected frames
COLMAP sparse reconstruction
camera parameters
```

Outputs:

```text
trained splat
checkpoints
training logs
preview renders
metrics
```

Use Nerfstudio / splatfacto as the first production training entrypoint.
This gives the pipeline a stable CLI-level training path while still using
gsplat underneath. Keep the wrapper backend-aware:

```yaml
training:
  backend: splatfacto  # splatfacto | raw_gsplat
  nerfstudio_dir: /workspace/opt/nerfstudio
  use_existing_colmap: true

colmap:
  binary: /workspace/opt/colmap-install/bin/colmap
  mapper: global_mapper
  use_nerfstudio_colmap: false
```

The future `raw_gsplat` backend is the right path if the project needs direct
access to newer gsplat features, semantic feature rendering, custom
segmentation-aware training, or lower-level memory/performance controls.

Training should be configurable:

```yaml
gsplat:
  max_steps: 30000
  downscale_factor: 1
  save_every: 1000
  eval_every: 1000
```

For development:

```yaml
max_steps: 5000
downscale_factor: 2
```

For final runs:

```yaml
max_steps: 30000+
downscale_factor: 1
```

---

### Stage 7: Export Artifacts

Each run should produce standardized final artifacts.
Make `.ply` the first canonical export because it is easier to debug, inspect,
archive, and convert. Add viewer-specific export, such as `.splat` or a
SuperSplat / PlayCanvas-compatible artifact, after the browser viewer choice is
fixed.

Output structure:

```text
final/
  scene.ply
  scene.splat
  pointcloud.ply
  cameras.json
  viewer_config.json
  preview.mp4
  metrics.json
```

The hosting/viewer layer should depend only on files under `final/`.

---

### Stage 8: Hosting

The hosted viewer should be separate from the processing pipeline.

Initial hosting model:

```text
static web app
  ↓
loads scene artifact from storage
  ↓
renders Gaussian splat in browser
```

The viewer should support:

- WASD navigation
- mouse look
- reset camera
- basic minimap or orientation aid
- rough two-click distance measurement
- optional labels/hotspots

Distance measurement should be presented as approximate, not survey-grade.

---

## 9. Segmentation Roadmap

Segmentation should be added after the base splat pipeline is stable.

Potential segmentation goals:

- segment walls/floors/ceilings
- segment furniture
- segment windows/mirrors
- mask private objects
- create clickable semantic regions
- enable rough room/object annotations

Possible approach:

```text
selected images
  ↓
SAM2 or other segmentation model
  ↓
2D masks per frame
  ↓
project masks using COLMAP cameras
  ↓
aggregate labels in 3D point cloud
  ↓
optionally transfer labels to splats
```

Segmentation should initially be optional and not part of the required reconstruction path.

Run location:

| Task | Location |
|---|---|
| simple mask review | local |
| SAM2 on many frames | cloud GPU or API |
| projecting masks to sparse points | local or cloud CPU |
| splat-level semantic refinement | cloud GPU |

Do not block the first product milestone on segmentation.

---

## 10. Recommended Repository Structure

```text
realestate-splat/
  README.md
  PROJECT_PLAN.md
  pyproject.toml
  requirements.txt

  configs/
    local_preprocess.yaml
    colmap_incremental.yaml
    colmap_global.yaml
    training_splatfacto_dev.yaml
    training_splatfacto_full.yaml

  scripts/
    preprocess_video.py
    make_capture_report.py
    upload_to_verda.py
    run_colmap.py
    run_training.py
    export_scene.py
    run_cloud_pipeline.py

  src/
    realestate_splat/
      preprocessing/
        extract_frames.py
        quality.py
        deduplicate.py
        frame_selector.py

      recommendation/
        gpu_recommender.py
        scene_estimator.py

      cloud/
        ssh.py
        rsync.py
        verda_paths.py

      reconstruction/
        colmap_commands.py
        colmap_outputs.py

      training/
        backends/
          splatfacto.py
          raw_gsplat.py

      segmentation/
        sam2_runner.py
        project_masks.py

      reporting/
        html_report.py
        contact_sheet.py

      hosting/
        upload_artifacts.py

  notebooks/
    01_frame_selection_debug.ipynb
    02_colmap_debug.ipynb
    03_pointcloud_view.ipynb
    04_segmentation_projection.ipynb

  runs/
    .gitkeep
```

---

## 11. Run Directory Contract

Every scene run should follow this structure:

```text
runs/house_001/
  frames_selected/

  masks/
    image_masks/

  colmap/
    database.db
    sparse/
    logs/

  gsplat/
    checkpoints/
    exports/
    logs/

  reports/
    capture_report.json
    capture_report.html
    frame_contact_sheet.jpg
    gpu_recommendation.json
    reconstruction_report.json

  run_config.json

  final/
    scene.ply
    scene.splat
    pointcloud.ply
    cameras.json
    viewer_config.json
    preview.mp4
    metrics.json
```

This directory contract is important because later automation can depend on it.
The local preprocessing stage should not write a raw extracted-frame cache by
default. Only final selected frames should be written to avoid unnecessary disk
use.

---

## 12. CLI Commands to Aim For

Local preprocessing:

```bash
python scripts/preprocess_video.py \
  --video data/raw/house_001.mp4 \
  --out runs/house_001 \
  --profile indoor_house
```

Upload to Verda:

```bash
python scripts/upload_to_verda.py \
  --run runs/house_001 \
  --host verda-a6000
```

Run COLMAP on Verda:

```bash
python scripts/run_colmap.py \
  --run /workspace/runs/house_001 \
  --config configs/training_splatfacto_dev.yaml
```

Run training on Verda:

```bash
python scripts/run_training.py \
  --run /workspace/runs/house_001 \
  --config configs/training_splatfacto_dev.yaml
```

Export final scene:

```bash
python scripts/export_scene.py \
  --run /workspace/runs/house_001
```

Eventually:

```bash
python scripts/run_cloud_pipeline.py \
  --run runs/house_001 \
  --host verda-a6000 \
  --mode global
```

---

## 13. Local vs Cloud Responsibilities

### Local Mac

Run locally:

- video frame extraction
- blur/quality scoring
- frame selection
- contact sheet generation
- capture report
- GPU recommendation
- optional lightweight preview app

Do not run locally:

- full gsplat training
- heavy segmentation over many frames
- large reconstruction jobs

### Verda GPU

Run on cloud GPU:

- COLMAP feature extraction
- COLMAP matching
- COLMAP mapper/global_mapper
- gsplat training
- heavy segmentation jobs
- artifact export

### Hosting

Run separately:

- static viewer app
- scene asset hosting
- shareable URLs
- analytics later

---

## 14. Verda Development Mode

During active development:

```text
Verda GPU instance
  +
persistent block volume
  +
manual SSH
  +
scripts
```

Use the block volume for:

```text
/workspace/envs/
/workspace/opt/colmap-install/
/workspace/runs/
/workspace/data/
```

The block volume is worth keeping during active testing because it avoids repeatedly reinstalling the Python environment and COLMAP.

---

## 15. Future Production Mode

Once the toolchain stabilizes, move to container-based processing.

Production target:

```text
container image:
  CUDA/PyTorch
  gsplat
  COLMAP CLI with global_mapper
  pipeline code

external storage:
  input frames/videos
  COLMAP outputs
  gsplat outputs
  final scene artifacts
```

Processing flow:

```text
upload video or frames
  ↓
create GPU job
  ↓
container pulls input
  ↓
container runs pipeline
  ↓
container uploads output
  ↓
viewer link is created
```

Avoid building a complex orchestration system before the pipeline is stable.

---

## 16. Capture Guidelines

Because the capture process is controlled initially, define a repeatable capture protocol.

Indoor capture rules:

- walk slowly
- avoid fast rotations
- avoid motion blur
- keep good overlap
- capture loops when possible
- capture room transitions carefully
- avoid filming directly into bright windows
- capture corners and doorways
- avoid purely forward hallway movement
- prefer normal wide phone camera first, not fisheye

Outdoor capture rules:

- orbit slowly
- keep object centered
- maintain overlap
- avoid extreme sky-only frames
- avoid strong reflections when possible
- prefer consistent exposure

Known problematic surfaces:

- mirrors
- windows
- glossy surfaces
- plain white walls
- moving objects
- people
- pets
- screens

These should eventually be detected or flagged in the capture report.

---

## 17. First Milestones

### Milestone 1: Local Preprocessing

Build:

```text
scripts/preprocess_video.py
```

Must output:

```text
frames_selected/
reports/capture_report.json
reports/capture_report.html
reports/frame_contact_sheet.jpg
reports/gpu_recommendation.json
run_config.json
```

Success criteria:

- can process one room video
- selects reasonable frames
- removes obvious blur/duplicates
- keeps gap-aware coverage across the video when possible
- creates a useful contact sheet
- gives a GPU recommendation

---

### Milestone 2: Manual Verda Pipeline

Build:

```text
scripts/run_colmap.py
scripts/run_training.py
scripts/export_scene.py
```

Success criteria:

- can upload selected frames manually
- can run COLMAP
- can run gsplat
- can export final artifact
- can view result locally or in a simple viewer

---

### Milestone 3: One-Command Cloud Run

Build:

```text
scripts/run_cloud_pipeline.py
```

Success criteria:

- local machine uploads selected frames
- remote GPU runs COLMAP + gsplat
- final artifacts are downloaded or uploaded
- logs are saved
- no notebook required

---

### Milestone 4: Viewer Prototype

Build browser viewer with:

- WASD navigation
- mouse look
- scene loading
- reset view
- rough distance measurement between two clicked points

Success criteria:

- one generated scene can be shared as a link
- viewer feels like a lightweight game walkthrough

---

### Milestone 5: Segmentation Prototype

Add optional segmentation path:

```text
selected images
  ↓
2D masks
  ↓
projection to point cloud
  ↓
semantic labels
```

Success criteria:

- can segment one object or room feature
- can project labels into 3D
- can visualize labeled points
- no requirement yet for production-quality semantic splats

---

## 18. Strict Priorities

Do first:

1. local preprocessing
2. frame selection report
3. COLMAP + training/export scripts
4. one reliable test scene
5. viewer prototype

Do later:

1. polished local UI
2. segmentation
3. full automation
4. arbitrary user uploads
5. advanced AR features

Avoid for now:

- Kubernetes
- complex queues
- full desktop app
- automatic segmentation before base reconstruction works
- supporting every camera type
- giant full-house scene as first test

---

## 19. Key Technical Risks

### Risk: Bad capture quality

Mitigation:

- controlled capture process
- capture guide
- quality report
- frame rejection

### Risk: Too many frames

Mitigation:

- frame selection
- image count targets
- GPU recommendation
- scene splitting

### Risk: GPU memory issues

Mitigation:

- conservative GPU recommendation
- downscale options
- checkpointing
- split scenes

### Risk: Mirrors/windows

Mitigation:

- capture guidance
- future masking/segmentation
- warning flags in report

### Risk: Pipeline becomes notebook-only

Mitigation:

- keep notebooks inspection-only
- scripts own the pipeline

### Risk: Premature UI work

Mitigation:

- build CLI first
- generate HTML reports
- wrap later with UI

---

## 20. Immediate Next Task

Start with the local preprocessing script.

Implement:

```text
scripts/preprocess_video.py
```

Minimum features:

- read video
- extract candidate frames at configurable FPS
- calculate blur score
- calculate brightness score
- reject obvious bad frames
- remove near-duplicates
- keep at least one acceptable or fallback frame per coverage window when possible
- save selected frames
- write capture_report.json
- write gpu_recommendation.json
- create contact sheet

This is the foundation for the entire system.
