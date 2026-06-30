#!/usr/bin/env bash
# Initialize a Verda runtime for Buildvision3D by running the three Verda setup scripts.
#
# Run this on the Verda GPU instance from the repository root:
#
#   cd /workspace/repo/realestate-splat
#   source verda/init_runtime.sh
#
# Use `source` so PATH, micromamba, Pixi, COLMAP, and CUDA build variables stay
# active in the current shell.
#
# What this does:
#
#   1. source verda/start_up_script.sh
#      Mounts /dev/vdb to /mnt/GaussianSplatVolume if needed, creates/repairs
#      /workspace -> /mnt/GaussianSplatVolume/workspace, and adds micromamba
#      plus COLMAP to PATH.
#
#   2. bash verda/install_colmap_runtime_deps.sh
#      Installs COLMAP/Nerfstudio runtime apt dependencies.
#
#   3. source verda/setup_pixi_env.sh
#      Adds Pixi/COLMAP paths and CUDA extension build variables.
#
# Optional:
#
#   SKIP_APT=1 source verda/init_runtime.sh
#   micromamba activate /workspace/envs/splat-dev
#
# `SKIP_APT=1` skips step 2 when apt dependencies are already installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_section() {
  printf '\n==> %s\n' "$1"
}

log_section "1/3 Mount workspace and expose micromamba/COLMAP"
# shellcheck source=verda/start_up_script.sh
source "${SCRIPT_DIR}/start_up_script.sh"

if [[ "${SKIP_APT:-0}" == "1" ]]; then
  log_section "2/3 Install COLMAP/Nerfstudio runtime dependencies"
  echo "  - Skipping apt install because SKIP_APT=1"
else
  log_section "2/3 Install COLMAP/Nerfstudio runtime dependencies"
  bash "${SCRIPT_DIR}/install_colmap_runtime_deps.sh"
fi

log_section "3/3 Set up Pixi/CUDA shell environment"
# shellcheck source=verda/setup_pixi_env.sh
source "${SCRIPT_DIR}/setup_pixi_env.sh"

log_section "Verda runtime init complete"
echo "  - Current folder: $(pwd)"
echo "  - micromamba: $(command -v micromamba || true)"
echo "  - pixi: $(command -v pixi || true)"
echo "  - colmap: $(command -v colmap || true)"
echo "  - Next: micromamba activate /workspace/envs/splat-dev"
