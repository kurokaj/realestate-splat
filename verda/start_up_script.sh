#!/usr/bin/env bash
set -euo pipefail

MOUNTPOINT="/mnt/GaussianSplatVolume"
DEVICE="/dev/vdb"
WORKSPACE="/workspace"
TARGET="${MOUNTPOINT}/workspace"

echo "==> Verda startup: mount volume and prepare /workspace"
echo "  - Mountpoint: ${MOUNTPOINT}"
echo "  - Device: ${DEVICE}"

mkdir -p "$MOUNTPOINT"

if ! mountpoint -q "$MOUNTPOINT"; then
  echo "  - Mounting ${DEVICE} at ${MOUNTPOINT}"
  mount "$DEVICE" "$MOUNTPOINT"
else
  echo "  - ${MOUNTPOINT} is already mounted; skipping mount"
fi

if [ ! -d "$TARGET" ]; then
  echo "ERROR: Expected workspace directory does not exist: ${TARGET}" >&2
  exit 1
fi

if [ -L "$WORKSPACE" ] && [ "$(readlink -f "$WORKSPACE")" = "$TARGET" ]; then
  echo "  - /workspace already points to ${TARGET}"
else
  if [ -e "$WORKSPACE" ] || [ -L "$WORKSPACE" ]; then
    if [ -L "$WORKSPACE" ]; then
      echo "  - Removing stale /workspace symlink"
      rm "$WORKSPACE"
    elif [ -d "$WORKSPACE" ] && [ -z "$(find "$WORKSPACE" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      echo "  - Removing empty /workspace directory"
      rmdir "$WORKSPACE"
    else
      echo "ERROR: ${WORKSPACE} exists and is not an empty directory or the expected symlink." >&2
      echo "Move it manually, then rerun this script." >&2
      exit 1
    fi
  fi
  echo "  - Linking ${WORKSPACE} -> ${TARGET}"
  ln -s "$TARGET" "$WORKSPACE"
fi

df -h "$WORKSPACE"

export PATH=/workspace/bin:$PATH

if [ -x /workspace/bin/micromamba ]; then
  eval "$(/workspace/bin/micromamba shell hook -s bash)"
  echo "  - micromamba found: $(command -v micromamba)"
else
  echo "  - micromamba not found at /workspace/bin/micromamba"
fi

export PATH=/workspace/opt/colmap-install/bin:$PATH
echo "  - Added COLMAP path: /workspace/opt/colmap-install/bin"
