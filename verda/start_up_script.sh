#!/usr/bin/env bash
set -euo pipefail

MOUNTPOINT="/mnt/GaussianSplatVolume"
DEVICE="/dev/vdb"
WORKSPACE="/workspace"
TARGET="${MOUNTPOINT}/workspace"

if ! mountpoint -q "$MOUNTPOINT"; then
  mount "$DEVICE" "$MOUNTPOINT"
fi

if [ ! -L "$WORKSPACE" ] || [ "$(readlink -f "$WORKSPACE")" != "$TARGET" ]; then
  rm -rf "$WORKSPACE"
  ln -s "$TARGET" "$WORKSPACE"
fi

df -h "$WORKSPACE"

# Add the micromamba to root PATH
export PATH=/workspace/bin:$PATH

if [ -x /workspace/bin/micromamba ]; then
  eval "$(/workspace/bin/micromamba shell hook -s bash)"
fi

export PATH=/workspace/opt/colmap-install/bin:$PATH