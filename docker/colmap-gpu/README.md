# COLMAP GPU Image

Canonical Dockerfile for the verified COLMAP GPU image.

The current Dockerfile is a multi-stage R2-runner recipe. The builder stage
keeps the heavy CUDA/C++ build environment, while the runtime stage contains
the installed COLMAP/Ceres artifacts, runtime libraries, Python, Git, and
`awscli`.

The repository is not baked into this image. Clone the repo branch at pod start
while the stage wrappers are still changing.

Build from this directory so Docker does not send local run data as context:

```bash
cd docker/colmap-gpu

export IMAGE_NAME="docker.io/blackjokuro/buildvision3d-colmap-gpu"
export IMAGE_TAG="cuda12.4-colmap-r2-runtime-onnx-cudnn-pycolmap-sm75-sm86-sm89-r1"
export CUDA_ARCHS="75;86;89"
export BUILD_JOBS=8
export COLMAP_REF="4.0.4"

docker build --network=host \
  --build-arg CUDA_ARCHS="$CUDA_ARCHS" \
  --build-arg BUILD_JOBS="$BUILD_JOBS" \
  --build-arg COLMAP_REF="$COLMAP_REF" \
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

The image verifies:

```text
ldd /opt/colmap-cuda/bin/colmap has no "not found" libraries
ldd /opt/colmap-cuda/lib/libonnxruntime_providers_cuda.so has no "not found" libraries
libcudnn.so.9 is present for ONNX Runtime CUDA provider
colmap -h works
colmap global_mapper -h works
colmap view_graph_calibrator -h works
same-source PyCOLMAP imports and exposes match_image_pairs/match_vocabtree
aws --version works
```

ALIKED/LightGlue ONNX model files are downloaded by COLMAP on first runtime
use. The Dockerfile has a note where future image-level model preloading can be
added if runtime GitHub downloads become flaky.

The Dockerfile removes apt package lists, deletes Ceres/COLMAP source trees
after installation in the builder, and does not copy those source trees into
the final runtime stage.
