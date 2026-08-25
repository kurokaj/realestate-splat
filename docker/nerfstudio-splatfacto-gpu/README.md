# Nerfstudio Splatfacto GPU Image

Canonical Dockerfile for the verified Nerfstudio/Pixi/Splatfacto GPU image.

The current Dockerfile is a conservative R2-runner cleanup recipe:

```text
keeps the proven CUDA devel + Pixi environment
adds awscli for R2 stage wrappers
keeps Git for branch cloning during wrapper iteration
does not bake this repository into the image
removes Nerfstudio .git metadata and common package/Python caches
verifies ns-train, ns-process-data, and ns-export gaussian-splat
re-verifies ns-train and ns-export after cleanup
```

Build from this directory so Docker does not send local run data as build
context:

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

Latest R2-clean image:

```text
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:cuda11.8-pixi-splatfacto-r2-clean-sm75-sm86-sm89-r2
compressed registry size: 11.72GB
digest: TBD after push/inspect
```

Previous verified image:

```text
docker.io/blackjokuro/buildvision3d-nerfstudio-splatfacto-gpu:cuda11.8-pixi-splatfacto-sm75-sm86-sm89-r1
sha256:f35527474f3e54d613542420a18a89500329aa33f9552750baa47123b2488243
```

Operational notes and smoke-test commands live in
`docs/docker_images.md`.
