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

# CUDA extension build settings for Nerfstudio/Pixi CUDA 11.8
if [ -x /usr/bin/gcc-11 ] && [ -x /usr/bin/g++-11 ]; then
  export CC=/usr/bin/gcc-11
  export CXX=/usr/bin/g++-11
  export CUDAHOSTCXX=/usr/bin/g++-11
fi

# Build tiny-cuda-nn / PyTorch CUDA extensions for common Verda GPU families:
#   7.0 = Tesla V100 / Volta
#   8.6 = RTX A6000 / Ampere
#   8.9 = RTX 6000 Ada / Ada
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-70;86;89}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.6;8.9}"

echo "TCNN_CUDA_ARCHITECTURES: ${TCNN_CUDA_ARCHITECTURES}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
