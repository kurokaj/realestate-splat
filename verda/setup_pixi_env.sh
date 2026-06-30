#!/usr/bin/env bash

# Run from the repository root on Verda:
#   source verda/setup_pixi_env.sh

export PATH=/workspace/bin:$PATH
export PATH=/workspace/pixi/bin:$PATH
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/workspace}"

if [ -x /workspace/bin/micromamba ]; then
  eval "$(/workspace/bin/micromamba shell hook -s bash)"
fi

export PATH=/workspace/opt/colmap-install/bin:$PATH

# CUDA extension build settings for RTX 6000 Ada + Nerfstudio/Pixi CUDA 11.8
if [ -x /usr/bin/gcc-11 ] && [ -x /usr/bin/g++-11 ]; then
  export CC=/usr/bin/gcc-11
  export CXX=/usr/bin/g++-11
  export CUDAHOSTCXX=/usr/bin/g++-11
fi

export TCNN_CUDA_ARCHITECTURES=89
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
