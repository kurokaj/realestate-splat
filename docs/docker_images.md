# Docker Images

Status: production-runtime container notes.

This document records image-specific details that are easy to forget when
moving from manual GPU testing to repeatable stage runners.

## COLMAP

The COLMAP image is the first dedicated GPU stage image. It is separate from
the future Gaussian splatting / Nerfstudio image.

Canonical Dockerfile:

```text
docker/colmap-gpu/Dockerfile
```

### Current image purpose

```text
Stage: COLMAP reconstruction
Input: selected frames + image_manifest.json
Output: COLMAP database, sparse model, reconstruction report, stage result
GPU provider target: RunPod first, Verda usable for building/testing
```

The image is built for the main RunPod target pool, not for Tesla V100.

```text
CUDA_ARCHS=75;86;89
CASPAR_ENABLED=ON
```

This covers:

```text
75  Turing / T4 fallback
86  RTX A6000, RTX A5000, A40, RTX 3090
89  L4, L40, L40S, RTX 4090, RTX 6000 Ada
```

Do not use this image for V100. If a V100-compatible image is needed later:

```text
CUDA_ARCHS=70;75;86;89
CASPAR_ENABLED=OFF
```

Caspar requires CUDA architecture 75 or newer.

### Build details to remember

The first verified image used:

```text
Base image: nvidia/cuda:12.4.1-devel-ubuntu22.04, single-stage
COLMAP: 4.2.0.dev0, commit cf7a8853, built 2026-07-21
Ceres: 2.3.0
CUDA: 12.4.1 in container
BLAS: OpenBLAS
cuDSS: CUDA 12 package, cudss-cuda-12
GUI: disabled
Caspar: enabled
```

Important dependency notes:

```text
Install awscli in the image revision that runs scripts/run_colmap_stage.py.
Install cudss-cuda-12, not generic cudss.
Generic cudss may install CUDA 13 cuDSS and cause libcublas.so.13 link failures.
Use OpenBLAS instead of MKL for simpler Ceres/SuiteSparse discovery.
Install newer CMake from Kitware because FAISS requires CMake >= 3.24.
Disable GUI with -DGUI_ENABLED=OFF to avoid Qt dependencies.
Install libopenexr-dev for OpenImageIO/OpenEXR CMake discovery.
Do not install libimath-dev on Ubuntu 22.04 because it conflicts with libilmbase-dev.
```

Build from the Dockerfile directory so Docker does not send local `runs/` data
as build context:

```bash
cd docker/colmap-gpu

export IMAGE_NAME="docker.io/blackjokuro/buildvision3d-colmap-gpu"
export IMAGE_TAG="cuda12.4-colmap-r2-runtime-sm75-sm86-sm89-r2"
export CUDA_ARCHS="75;86;89"
export BUILD_JOBS=8

docker build --network=host \
  --build-arg CUDA_ARCHS="$CUDA_ARCHS" \
  --build-arg BUILD_JOBS="$BUILD_JOBS" \
  -t "$IMAGE_NAME:$IMAGE_TAG" \
  .
```

The current Dockerfile is a multi-stage R2-runner recipe:

```text
builder stage:
  nvidia/cuda:12.4.1-devel-ubuntu22.04
  compiler/build tools
  Ceres and COLMAP source trees

runtime stage:
  nvidia/cuda:12.4.1-runtime-ubuntu22.04
  awscli, git, Python for R2 stage wrappers
  copied /opt/ceres-cuda and /opt/colmap-cuda
  no repository code baked in
```

The Dockerfile removes apt caches, deletes Ceres/COLMAP source trees after
install in the builder, and does not copy those source trees into the final
runtime stage. On the Verda build, `docker image inspect` reported about
2.39GB for this runtime tag. Build will fail if
`ldd /opt/colmap-cuda/bin/colmap` reports a missing shared library.

### Option namespaces

COLMAP option names vary by build. This image uses the newer feature option
namespaces.

Feature extraction:

```bash
--FeatureExtraction.use_gpu 1
--FeatureExtraction.gpu_index 0
```

Do not use these old names with this image:

```text
--SiftExtraction.gpu_index
```

Feature matching:

```bash
--FeatureMatching.use_gpu 1
--FeatureMatching.gpu_index 0
```

Do not use these old names with this image:

```text
--SiftMatching.gpu_index
```

Incremental mapper:

```bash
--Mapper.ba_use_gpu 1
--Mapper.ba_gpu_index 0
```

Do not pass standalone bundle-adjuster options to `colmap mapper`:

```text
--BundleAdjustmentCeres.use_gpu
--BundleAdjustmentCeres.gpu_index
--BundleAdjustmentCeres.min_num_images_gpu_solver
```

Those are accepted by `colmap bundle_adjuster`, not by `colmap mapper`.

Global mapper:

```bash
--GlobalMapper.gp_use_gpu 1
--GlobalMapper.gp_gpu_index 0
--GlobalMapper.ba_ceres_use_gpu 1
--GlobalMapper.ba_gpu_index 0
```

The global path should still run `view_graph_calibrator` first:

```text
database.db
  -> copy to database_global.db
  -> view_graph_calibrator on database_global.db
  -> global_mapper using database_global.db
```

### R2-backed stage wrapper

The repo-side stage wrapper is:

```text
scripts/run_colmap_stage.py
```

For the current stage-runner recipe, mount or clone the repo into the GPU
runtime and run the wrapper from the repository root. The R2-runtime image has
the verified COLMAP binary, `awscli`, `git`, and Python; the repo supplies the
stage script.

The first pushed COLMAP image digest was verified for reconstruction before the
R2 wrapper existed. The current `cuda12.4-colmap-r2-runtime-sm75-sm86-sm89-r2`
tag is the first R2-backed runner image.

During wrapper development, do not rebuild the image just to iterate on Python.
Use the pushed COLMAP image as the GPU runtime, clone the repo branch, and run
the wrapper from the clone. A later locked production image may bake a tested
repo commit, but that is intentionally deferred while the Python wrappers are
changing quickly.

Default wrapper behavior:

```text
Input:  preprocess/current/
        frames_selected/
        image_manifest.json
        capture_report.json
        preprocess_summary.json

Local adapter shape:
        frames_selected/
        reports/image_manifest.json
        reports/capture_report.json
        reports/preprocess_summary.json

Output: colmap/current/
        database.db
        database_global.db, for global mapper runs
        sparse/
        sparse_txt/
        reconstruction_report.json
        stage_result.json

History: colmap/runs/<stage_run_id>/
        reconstruction_report.json
        stage_result.json
```

Successful runs do not upload terminal logs. Failed runs upload
`colmap/current/logs/` plus `stage_result.json`.

Command shape:

```bash
python3 scripts/run_colmap_stage.py \
  --project-id car_single_smoke \
  --input-uri "r2://$R2_BUCKET/projects/car_single_smoke/preprocess/current" \
  --output-uri "r2://$R2_BUCKET/projects/car_single_smoke/colmap" \
  --endpoint-url "$R2_ENDPOINT" \
  --mode global \
  --matcher exhaustive \
  --colmap-bin /opt/colmap-cuda/bin/colmap
```

The wrapper defaults to the verified global mapper route and passes through the
existing `scripts/run_colmap.py` options. Use repeated `--feature-option`,
`--matcher-option`, `--view-graph-calibrator-option`, `--mapper-option`, or
raw `--colmap-arg` values for experiments.

### Manual R2 smoke test without rebuilding

Use this path while `scripts/run_colmap_stage.py` is still changing.

Start a GPU pod/container from:

```text
docker.io/blackjokuro/buildvision3d-colmap-gpu:latest-dev
```

Then inside the pod:

```bash
apt-get update
apt-get install -y --no-install-recommends awscli git ca-certificates

git clone --branch production-runtime-roadmap <REPO_URL> /workspace/Buildvision3D
cd /workspace/Buildvision3D

export AWS_ACCESS_KEY_ID="<r2-access-key-id>"
export AWS_SECRET_ACCESS_KEY="<r2-secret-access-key>"
export AWS_DEFAULT_REGION="auto"
export R2_ENDPOINT="<r2-endpoint-url>"
export R2_BUCKET="buildvision3d-pipeline"

python3 scripts/run_colmap_stage.py \
  --project-id car_single_smoke \
  --input-uri "r2://$R2_BUCKET/projects/car_single_smoke/preprocess/current" \
  --output-uri "r2://$R2_BUCKET/projects/car_single_smoke/colmap" \
  --endpoint-url "$R2_ENDPOINT" \
  --mode global \
  --matcher exhaustive \
  --colmap-bin /opt/colmap-cuda/bin/colmap
```

Quick result checks:

```bash
aws --endpoint-url "$R2_ENDPOINT" s3 ls \
  "s3://$R2_BUCKET/projects/car_single_smoke/colmap/current/"

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  "s3://$R2_BUCKET/projects/car_single_smoke/colmap/current/stage_result.json" -

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  "s3://$R2_BUCKET/projects/car_single_smoke/colmap/current/reconstruction_report.json" -
```

Expected current outputs:

```text
database.db
database_global.db
sparse/
sparse_txt/
reconstruction_report.json
stage_result.json
```

If the stage fails, check:

```text
colmap/current/stage_result.json
colmap/current/logs/
```

### Verification commands

Basic image verification:

```bash
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" nvidia-smi
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" colmap -h
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" colmap feature_extractor -h
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" colmap exhaustive_matcher -h
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" colmap mapper -h
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" colmap global_mapper -h
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" colmap view_graph_calibrator -h
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" colmap bundle_adjuster -h
```

Linkage verification:

```bash
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" \
  bash -lc 'ldd "$(which colmap)" | grep -Ei "ceres|cuda|cudss|cusolver|cusparse|cublas|suitesparse|openblas"'
```

Good signs:

```text
libcudart.so.12
libcusolver.so.11
libcublas.so.12
libcusparse.so.12
libcudss.so.0
libopenblas.so.0
```

Bad sign:

```text
libcublas.so.13
```

That means CUDA 13 cuDSS was installed into a CUDA 12 image.

Warning grep after smoke tests:

```bash
grep -Rin "compiled without cuDSS\|falling back to CPU\|Requested to use GPU" /workspace/colmap-smoke 2>/dev/null
```

No output is good.

### Smoke-test command shape

Feature extraction:

```bash
docker run --rm --gpus all \
  -v /workspace/colmap-smoke:/workspace/colmap-smoke \
  "$IMAGE_NAME:$IMAGE_TAG" \
  colmap feature_extractor \
    --database_path /workspace/colmap-smoke/database.db \
    --image_path /workspace/colmap-smoke/images \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 1 \
    --FeatureExtraction.gpu_index 0
```

Exhaustive matching:

```bash
docker run --rm --gpus all \
  -v /workspace/colmap-smoke:/workspace/colmap-smoke \
  "$IMAGE_NAME:$IMAGE_TAG" \
  colmap exhaustive_matcher \
    --database_path /workspace/colmap-smoke/database.db \
    --FeatureMatching.use_gpu 1 \
    --FeatureMatching.gpu_index 0
```

Incremental mapper:

```bash
docker run --rm --gpus all \
  -v /workspace/colmap-smoke:/workspace/colmap-smoke \
  "$IMAGE_NAME:$IMAGE_TAG" \
  colmap mapper \
    --database_path /workspace/colmap-smoke/database.db \
    --image_path /workspace/colmap-smoke/images \
    --output_path /workspace/colmap-smoke/sparse \
    --Mapper.ba_use_gpu 1 \
    --Mapper.ba_gpu_index 0
```

Global mapper preparation:

```bash
cp /workspace/colmap-smoke/database.db /workspace/colmap-smoke/database_global.db

docker run --rm --gpus all \
  -v /workspace/colmap-smoke:/workspace/colmap-smoke \
  "$IMAGE_NAME:$IMAGE_TAG" \
  colmap view_graph_calibrator \
    --database_path /workspace/colmap-smoke/database_global.db
```

Global mapper:

```bash
docker run --rm --gpus all \
  -v /workspace/colmap-smoke:/workspace/colmap-smoke \
  "$IMAGE_NAME:$IMAGE_TAG" \
  colmap global_mapper \
    --database_path /workspace/colmap-smoke/database_global.db \
    --image_path /workspace/colmap-smoke/images \
    --output_path /workspace/colmap-smoke/sparse_global \
    --GlobalMapper.gp_use_gpu 1 \
    --GlobalMapper.gp_gpu_index 0 \
    --GlobalMapper.ba_ceres_use_gpu 1 \
    --GlobalMapper.ba_gpu_index 0
```

### Verified smoke result

Verified on 2026-07-21 with `runs/car_single/frames_selected` copied to a
Verda RTX A6000 VM.

Global mapper output contained:

```text
/workspace/colmap-smoke/sparse_global/0/cameras.bin
/workspace/colmap-smoke/sparse_global/0/frames.bin
/workspace/colmap-smoke/sparse_global/0/images.bin
/workspace/colmap-smoke/sparse_global/0/points3D.bin
/workspace/colmap-smoke/sparse_global/0/rigs.bin
/workspace/colmap-smoke/sparse_global/project.ini
```

The cuDSS/fallback warning grep returned no output.

Observed global mapper notes:

```text
325 images loaded
global positioning converged
iterative bundle adjustment completed
reconstruction done in 324.224 seconds
```

Incremental mapper was slower than global mapper on this smoke input. Both
showed GPU activity in bursts. Short 0% `nvidia-smi` readings during mapper are
not necessarily a problem because some stages are CPU-bound and GPU work may be
brief.

## Image Version Log

### 2026-07-21: COLMAP GPU CUDA 12.4, Caspar, RunPod target

Registry:

```text
docker.io
```

Repository:

```text
blackjokuro/buildvision3d-colmap-gpu
```

Tags:

```text
docker.io/blackjokuro/buildvision3d-colmap-gpu:cuda12.4-colmap-global-caspar-sm75-sm86-sm89-r1
docker.io/blackjokuro/buildvision3d-colmap-gpu:latest-dev
```

Digest:

```text
sha256:41ff37f24dbce2064147436c739f7711b997e8c130f2a26d8a2d2b67db240e4f
```

Build/verification environment:

```text
Verda VM
RTX A6000
48GB VRAM
10 vCPU
60GB RAM
200GiB OS volume
Ubuntu 22.04
CUDA 12.4
Docker
```

Verification status:

```text
Pushed to Docker Hub.
Docker GPU passthrough verified.
COLMAP with CUDA verified.
global_mapper verified.
view_graph_calibrator verified.
Ceres/CUDA/cuDSS linkage verified.
Caspar options present.
car_single smoke reconstruction completed.
```

R2 wrapper status:

```text
Original r1 digest was built before scripts/run_colmap_stage.py existed.
Current r2 runtime tag includes awscli/git/Python for R2 stage execution.
Repo code is still cloned at pod startup for faster wrapper iteration.
```

## Image Startup / Pull Optimization Backlog

Cold pod startup can be billed while the image is downloading/extracting. A
10-minute cold pull is acceptable for smoke testing, but too slow for the
future automated pipeline. Revisit these items when rebuilding images for the
R2 runner tags.

General actions:

```text
Keep awscli/git baked into R2 runner images so pod startup does not run apt-get.
Keep repo code cloned at pod startup until the wrappers settle.
Use registry auth in RunPod to avoid Docker Hub shared-IP throttling.
Prefer pinned tags/digests, but keep latest-dev only for manual development.
Keep image build contexts tiny; never send runs/ or local data to Docker.
Add .dockerignore files beside image builds if needed.
Measure cold pull + extract time separately from stage runtime in smoke notes.
```

Build-cache actions:

```text
Set up docker buildx registry cache for future rebuilds:
  --cache-to type=registry,...
  --cache-from type=registry,...

Do not rely on disposable GPU VM local Docker cache.
Structure Dockerfiles so Python stage code is copied near the end.
Python-only changes should then rebuild only the final layers.
```

COLMAP image actions:

```text
awscli/git/Python are included in the current r2 runtime image.
Optionally bake scripts/run_colmap_stage.py, scripts/run_colmap.py, and src/ later.
Keep SSH out of the production image; use a separate debug template/tag if needed.
The current r2 image is multi-stage/runtime-only and was pushed at about 2.2GB compressed.
```

Nerfstudio image actions:

```text
High priority: reduce image size after the R2 training smoke succeeds.
Move from CUDA devel base to CUDA runtime base with a multi-stage build.
Clean Pixi/conda/pip caches before the final layer.
Avoid leaving compiler/build caches in the runtime image.
Check whether /workspace/opt/nerfstudio contains unused git/build artifacts.
Keep /root/.cache/torch out of the image unless intentionally prewarming LPIPS.
Mount /root/.cache/torch on reusable debug pods to avoid redownloading AlexNet.
Consider a split build image + runtime image for tiny-cuda-nn/gsplat artifacts.
awscli/git are already included in the current R2-clean runner.
Optionally bake stage scripts into a locked production image once the wrapper settles.
```

## Nerfstudio / Splatfacto

The Nerfstudio/Splatfacto image is the dedicated Gaussian splatting training
image. It is separate from the COLMAP image because Nerfstudio has a different
CUDA/PyTorch/Pixi dependency profile.

Canonical Dockerfile:

```text
docker/nerfstudio-splatfacto-gpu/Dockerfile
```

### Goal

```text
Stage: Gaussian splatting training
Input: selected frames, image_manifest.json, COLMAP sparse model
Output: checkpoints, training metrics, exported splat artifacts, stage result
GPU provider target: RunPod first
```

The image should reduce the cold start of the old Verda-style runtime by baking
the slow dependency installation and CUDA extension compilation into the image.

### Current image purpose

```text
Stage: Gaussian splatting training
Input: selected frames, Nerfstudio transforms/data, COLMAP sparse model
Output: checkpoints, training metrics, exported splat artifacts, stage result
GPU provider target: RunPod first, Verda usable for building/testing
```

The first image uses CUDA 11.8 because the Pixi/Nerfstudio path is CUDA
11.8-oriented and tiny-cuda-nn/gsplat are sensitive to CUDA/PyTorch mismatches.
This is intentionally different from the COLMAP image.

```text
COLMAP image:      CUDA 12.4
Nerfstudio image:  CUDA 11.8
```

The image is built for the same main RunPod target GPU pool as COLMAP:

```text
TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9"
TCNN_CUDA_ARCHITECTURES="75;86;89"
```

Do not include V100 support in the first RunPod-oriented training image unless
we intentionally make a separate legacy Verda image.

### What to bake

The verified image uses:

```text
Base image: nvidia/cuda:11.8.0-devel-ubuntu22.04
Nerfstudio: v1.1.4
Torch CUDA: cu118 path, verified torch.version.cuda = 11.8
Pixi: installed at /workspace/pixi
Nerfstudio checkout: /workspace/opt/nerfstudio
gsplat: import verified
tiny-cuda-nn: import verified with GPU available
```

Important dependency/runtime notes:

```text
Install awscli in the R2 runner revision so scripts/run_training_stage.py can sync R2.
Keep Git in the image while stage wrappers are cloned at pod startup.
Do not verify tinycudann during docker build; docker build has no GPU.
Verify torch.cuda, tinycudann, and gsplat with docker run --gpus all.
Pixi manifest deprecation warnings from Nerfstudio v1.1.4 are acceptable for now.
The first ns-train run may pause during LPIPS/AlexNet download and image caching/undistortion.
Mount /root/.cache/torch if reusing the same VM/pod to avoid redownloading AlexNet weights.
```

Build from the Dockerfile directory:

```bash
cd docker/nerfstudio-splatfacto-gpu

export IMAGE_NAME="docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu"
export IMAGE_TAG="cuda11.8-pixi-splatfacto-r2-clean-sm75-sm86-sm89-r2"
export NS_REF="v1.1.4"
export BUILD_JOBS=8

docker build --network=host \
  --build-arg NS_REF="$NS_REF" \
  --build-arg BUILD_JOBS="$BUILD_JOBS" \
  -t "$IMAGE_NAME:$IMAGE_TAG" \
  .
```

The current R2-clean Dockerfile keeps the proven single-stage CUDA/Pixi shape
and removes low-risk leftovers:

```text
Nerfstudio .git metadata
pip/pixi/torch/nv caches
Pixi package cache at .pixi/pkgs
Python __pycache__ and .pyc files
apt package lists
```

Do not remove `/workspace/opt/nerfstudio/.pixi/envs/default`; that is the
actual runtime environment.

The Dockerfile runs a second lightweight `pixi run` verification after cleanup.
That catches the important failure mode where deleting caches accidentally
removes something the runtime environment still needs.

### Verification commands

The image should verify:

```bash
docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" nvidia-smi

docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" \
  /workspace/pixi/bin/pixi run --manifest-path /workspace/opt/nerfstudio/pixi.toml \
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" \
  /workspace/pixi/bin/pixi run --manifest-path /workspace/opt/nerfstudio/pixi.toml \
  python -c "import tinycudann; print('tinycudann OK')"

docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" \
  /workspace/pixi/bin/pixi run --manifest-path /workspace/opt/nerfstudio/pixi.toml \
  python -c "import gsplat; print('gsplat OK')"

docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" \
  /workspace/pixi/bin/pixi run --manifest-path /workspace/opt/nerfstudio/pixi.toml \
  ns-train splatfacto --help

docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" \
  /workspace/pixi/bin/pixi run --manifest-path /workspace/opt/nerfstudio/pixi.toml \
  ns-export gaussian-splat --help

docker run --rm --gpus all "$IMAGE_NAME:$IMAGE_TAG" aws --version
```

### Smoke-test command shape

The verified smoke test used `runs/parkkihalli_dome_gap_aware` because it had
`frames_selected/` and `colmap/sparse_txt/` available locally.

Prepare data using the project text-COLMAP exporter:

```bash
docker run --rm --gpus all --ipc=host \
  -v /workspace/training-smoke:/workspace/training-smoke \
  -v /workspace/buildvision3d-scripts:/workspace/buildvision3d-scripts \
  "$IMAGE_NAME:$IMAGE_TAG" \
  /workspace/pixi/bin/pixi run --manifest-path /workspace/opt/nerfstudio/pixi.toml \
  python /workspace/buildvision3d-scripts/scripts/prepare_nerfstudio_from_colmap.py \
    --run /workspace/training-smoke \
    --frames-dir /workspace/training-smoke/frames_selected \
    --data-dir /workspace/training-smoke/nerfstudio \
    --colmap-model-dir /workspace/training-smoke/colmap/sparse_txt \
    --num-downscales 1 \
    --overwrite
```

Short training smoke:

```bash
docker run --rm --gpus all --ipc=host \
  -v /workspace/training-smoke:/workspace/training-smoke \
  -v /workspace/cache/torch:/root/.cache/torch \
  -v /workspace/cache/nv:/root/.nv \
  "$IMAGE_NAME:$IMAGE_TAG" \
  /workspace/pixi/bin/pixi run --manifest-path /workspace/opt/nerfstudio/pixi.toml \
  ns-train splatfacto \
    --data=/workspace/training-smoke/nerfstudio \
    --output-dir=/workspace/training-smoke/gsplat/outputs \
    --experiment-name=smoke \
    --max-num-iterations=100 \
    --steps-per-save=50 \
    --steps-per-eval-batch=50 \
    --steps-per-eval-image=50 \
    --viewer.quit-on-train-completion=True
```

### Stage behavior to preserve later

Training should not rerun COLMAP. It must consume the COLMAP output from the
previous stage:

```text
training.use_existing_colmap = true
colmap.use_nerfstudio_colmap = false
```

The image should support:

```text
short smoke run
full training run
checkpoint every 10-15 minutes
resume from latest checkpoint
metrics upload
final artifact export
failure stage_result.json
```

Nerfstudio image caching/undistortion is scene-specific runtime work. Do not
try to solve that in the image. Later, store prepared `nerfstudio/` data as a
stage artifact so repeated training runs can skip it when inputs and downscale
settings have not changed.

### R2-backed stage wrapper

The repo-side stage wrapper is:

```text
scripts/run_training_stage.py
```

During wrapper development, use the pushed Nerfstudio image as the GPU runtime,
clone the repo branch at pod startup, and run the wrapper from the clone. The
R2-clean image includes `awscli` and `git`; the repo is intentionally not baked
into the image while wrapper code is still changing quickly.

Default wrapper behavior:

```text
Inputs:
  preprocess/current/
    frames_selected/
    image_manifest.json
    capture_report.json
    preprocess_summary.json

  colmap/current/
    sparse_txt/
    sparse/
    reconstruction_report.json
    stage_result.json

Local adapter shape:
  frames_selected/
  colmap/sparse_txt/
  colmap/sparse/
  reports/reconstruction_report.json

Process:
  scripts/prepare_nerfstudio_from_colmap.py
  pixi run ns-train splatfacto
  pixi run ns-export gaussian-splat

Output:
  training/current/
    outputs/
    exports/splat.ply
    nerfstudio/transforms.json
    training_summary.json
    stage_result.json

History:
  training/runs/<stage_run_id>/
    training_summary.json
    stage_result.json
```

Successful runs do not upload terminal logs. Heavy artifacts are stored only in
`training/current/`, not duplicated into history runs. Failed runs upload
`training/current/logs/` and, if present, `training/current/outputs_partial/`.

### Manual R2 smoke test without rebuilding

Start a GPU pod/container from:

```text
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:latest-dev
```

For Web Terminal development, keep the container alive:

```bash
bash -lc 'sleep infinity'
```

Then inside the pod:

```bash
cd /workspace
git clone --branch production-runtime-roadmap https://github.com/kurokaj/realestate-splat.git Buildvision3D
cd /workspace/Buildvision3D

export AWS_ACCESS_KEY_ID="<r2-access-key-id>"
export AWS_SECRET_ACCESS_KEY="<r2-secret-access-key>"
export AWS_DEFAULT_REGION="auto"
export R2_ENDPOINT="<r2-endpoint-url>"
export R2_BUCKET="buildvision3d-pipeline"

python3 scripts/run_training_stage.py \
  --project-id car_single_smoke \
  --preprocess-uri "r2://$R2_BUCKET/projects/car_single_smoke/preprocess/current" \
  --colmap-uri "r2://$R2_BUCKET/projects/car_single_smoke/colmap/current" \
  --output-uri "r2://$R2_BUCKET/projects/car_single_smoke/training" \
  --endpoint-url "$R2_ENDPOINT" \
  --max-steps 100 \
  --save-every 50 \
  --eval-every 50 \
  --num-downscales 1
```

The wrapper runs `prepare_nerfstudio_from_colmap.py` through Pixi by default.
This matters because system `/usr/bin/python3` may not have Pillow/OpenCV, while
the Nerfstudio/Pixi environment does. If testing an older clone of the wrapper
that fails with `Pillow or OpenCV is required for --num-downscales > 0`, either
pull the latest branch or rerun with `--num-downscales 0` as a temporary smoke
test workaround.

Quick result checks:

```bash
aws --endpoint-url "$R2_ENDPOINT" s3 ls \
  "s3://$R2_BUCKET/projects/car_single_smoke/training/current/"

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  "s3://$R2_BUCKET/projects/car_single_smoke/training/current/stage_result.json" -

aws --endpoint-url "$R2_ENDPOINT" s3 cp \
  "s3://$R2_BUCKET/projects/car_single_smoke/training/current/training_summary.json" -

aws --endpoint-url "$R2_ENDPOINT" s3 ls \
  "s3://$R2_BUCKET/projects/car_single_smoke/training/current/exports/"
```

### Verified smoke result

Verified on 2026-07-29 with `runs/parkkihalli_dome_gap_aware` copied to a
Verda RTX 6000 Ada VM.

Smoke outputs included:

```text
/workspace/training-smoke/gsplat/outputs/smoke/splatfacto/2026-07-29_190853/config.yml
/workspace/training-smoke/gsplat/outputs/smoke/splatfacto/2026-07-29_190853/dataparser_transforms.json
/workspace/training-smoke/gsplat/outputs/smoke_timed/splatfacto/2026-07-29_191903/config.yml
/workspace/training-smoke/gsplat/outputs/smoke_timed/splatfacto/2026-07-29_191903/dataparser_transforms.json
```

R2 wrapper smoke verified on 2026-08-01 with `car_single_smoke`:

```text
Stage run id: training_20260801T094215
Input preprocess: r2://buildvision3d-pipeline/projects/car_single_smoke/preprocess/current/
Input COLMAP:     r2://buildvision3d-pipeline/projects/car_single_smoke/colmap/current/
Current output:   r2://buildvision3d-pipeline/projects/car_single_smoke/training/current/
History output:   r2://buildvision3d-pipeline/projects/car_single_smoke/training/runs/training_20260801T094215/
```

Export substep smoke verified after adding `ns-export gaussian-splat`:

```text
Canonical current export: r2://buildvision3d-pipeline/projects/car_single_smoke/training/current/exports/splat.ply
History remains metadata-only; exported PLY is not duplicated under training/runs/<stage_run_id>/.
```

## Nerfstudio Image Version Log

### 2026-08-02: Nerfstudio Splatfacto R2-clean runner

Registry:

```text
docker.io
```

Repository:

```text
blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu
```

Tags:

```text
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:cuda11.8-pixi-splatfacto-r2-clean-sm75-sm86-sm89-r2
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:latest-dev
```

Digest:

```text
TBD: paste from docker inspect --format='{{index .RepoDigests 0}}' after push.
```

Size:

```text
Compressed registry size: 11.72GB
Previous compressed registry size: 11.79GB
Local docker image inspect size from Verda A100 build: 12584508533 bytes
```

Build/verification environment:

```text
Verda VM
A100 40GB
Container CUDA 11.8
Docker
```

Changes from r1:

```text
Added awscli for R2-backed training stage execution.
Kept git for fast repo clone at pod startup.
Removed Nerfstudio .git metadata.
Removed common pip/pixi/torch/nv caches.
Removed Pixi package cache while preserving .pixi/envs/default.
Removed Python __pycache__ and .pyc files.
Added ns-export gaussian-splat verification.
Added post-cleanup pixi verification.
```

Verification status:

```text
Docker GPU passthrough verified on A100.
awscli verified.
torch.cuda verified: torch 2.2.2, CUDA visible, torch.version.cuda 11.8.
tiny-cuda-nn import verified.
gsplat import verified.
ns-export gaussian-splat help verified.
tiny-cuda-nn warned about A100 compute capability 80 because this image targets 75/86/89; acceptable for the RunPod target pool.
```

### 2026-07-29: Nerfstudio Splatfacto CUDA 11.8, Pixi, RunPod target

Registry:

```text
docker.io
```

Repository:

```text
blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu
```

Tags:

```text
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:cuda11.8-pixi-splatfacto-sm75-sm86-sm89-r1
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:latest-dev
```

Digest:

```text
sha256:f35527474f3e54d613542420a18a89500329aa33f9552750baa47123b2488243
```

Build/verification environment:

```text
Verda VM
RTX 6000 Ada
Ubuntu 22.04
Host CUDA 12.8
Container CUDA 11.8
Docker
```

Verification status:

```text
Pushed to Docker Hub.
Docker GPU passthrough verified.
torch.cuda verified.
tiny-cuda-nn import verified.
gsplat import verified.
ns-train splatfacto help verified.
parkkihalli_dome_gap_aware smoke training completed.
```
