# COLMAP GPU Image

Canonical Dockerfile for the verified COLMAP GPU image.

Build from this directory so Docker does not send local run data as build
context:

```bash
cd docker/colmap-gpu

export IMAGE_NAME="docker.io/blackjokuro/buildvision3d-colmap-gpu"
export IMAGE_TAG="cuda12.4-colmap-global-caspar-sm75-sm86-sm89-r1"
export CUDA_ARCHS="75;86;89"
export BUILD_JOBS=8

docker build --no-cache --network=host \
  --build-arg CUDA_ARCHS="$CUDA_ARCHS" \
  --build-arg BUILD_JOBS="$BUILD_JOBS" \
  -t "$IMAGE_NAME:$IMAGE_TAG" \
  .
```

Verified image:

```text
docker.io/blackjokuro/buildvision3d-colmap-gpu:cuda12.4-colmap-global-caspar-sm75-sm86-sm89-r1
sha256:41ff37f24dbce2064147436c739f7711b997e8c130f2a26d8a2d2b67db240e4f
```

Operational notes and smoke-test commands live in
`docs/docker_images.md`.

The R2-backed stage wrapper lives in:

```text
scripts/run_colmap_stage.py
```

For the current image revision, clone or mount this repository into the GPU
runtime and run the wrapper from the repository root with
`--colmap-bin /opt/colmap-cuda/bin/colmap`.

The stage-runner image revision must include `awscli`, because the storage
helper uses `aws s3 sync` for R2.
