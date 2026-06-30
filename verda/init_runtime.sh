#!/usr/bin/env bash
# Initialize a fresh Verda runtime for Buildvision3D.
#
# First bootstrap the persistent volume manually, because this repository lives
# under /workspace and is not available until the volume is mounted:
#
#   mkdir -p /mnt/GaussianSplatVolume
#   mount /dev/vdb /mnt/GaussianSplatVolume
#   rm -rf /workspace
#   ln -s /mnt/GaussianSplatVolume/workspace /workspace
#   df -h /workspace
#
# Then run this on the Verda GPU instance from the repository root:
#
#   cd /workspace/repo/realestate-splat
#   source verda/init_runtime.sh
#
# Use `source` when you want the PATH, micromamba hook, COLMAP path, Pixi path,
# and CUDA build environment variables to stay active in your current shell.
# If you only want to check/install runtime dependencies, this also works:
#
#   cd /workspace/repo/realestate-splat
#   bash verda/init_runtime.sh
#
# Optional:
#
#   SKIP_APT=1 source verda/init_runtime.sh
#
# `SKIP_APT=1` skips apt dependency installation when you know the VM image
# already has the COLMAP/Nerfstudio runtime libraries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="/workspace"
SETUP_ENV_TARGET="${WORKSPACE}/setup_env.sh"

log_section() {
  printf '\n==> %s\n' "$1"
}

log_step() {
  printf '  - %s\n' "$1"
}

run_script() {
  local label="$1"
  local script_path="$2"

  log_section "$label"
  log_step "Running ${script_path}"
  bash "$script_path"
}

require_workspace_ready() {
  if [[ ! -e "${WORKSPACE}" ]]; then
    log_section "Workspace is not mounted"
    log_step "${WORKSPACE} does not exist yet."
    log_step "Run the manual mount bootstrap from the comment at the top of this script, then rerun:"
    log_step "cd /workspace/repo/realestate-splat"
    log_step "source verda/init_runtime.sh"
    exit 1
  fi

  if [[ ! -d "${WORKSPACE}" ]]; then
    log_section "Workspace is not a directory"
    log_step "${WORKSPACE} exists but is not a usable directory."
    log_step "Check the /workspace symlink and mounted Verda block volume."
    exit 1
  fi
}

log_section "Buildvision3D Verda runtime initialization"
log_step "Repository root: ${REPO_ROOT}"
log_step "Script directory: ${SCRIPT_DIR}"
log_step "Expected workspace: ${WORKSPACE}"
require_workspace_ready

if [[ "$(id -u)" -ne 0 ]]; then
  log_section "Warning"
  log_step "This script is expected to run as root on the Verda instance."
  log_step "Installing apt packages may fail without root privileges."
fi

if [[ "${SKIP_APT:-0}" == "1" ]]; then
  log_section "1/3 Install COLMAP/Nerfstudio runtime dependencies"
  log_step "Skipping apt install because SKIP_APT=1."
else
  run_script "1/3 Install COLMAP/Nerfstudio runtime dependencies" "${SCRIPT_DIR}/install_colmap_runtime_deps.sh"
fi

log_section "2/3 Install reusable shell environment helper"
log_step "Copying ${SCRIPT_DIR}/setup_pixi_env.sh to ${SETUP_ENV_TARGET}"
cp "${SCRIPT_DIR}/setup_pixi_env.sh" "${SETUP_ENV_TARGET}"
chmod +x "${SETUP_ENV_TARGET}"

log_section "3/3 Source Verda shell environment"
log_step "Sourcing ${SETUP_ENV_TARGET}"
# shellcheck source=/workspace/setup_env.sh
source "${SETUP_ENV_TARGET}"

log_section "Runtime verification"
log_step "PATH includes: /workspace/bin, /workspace/pixi/bin, /workspace/opt/colmap-install/bin"

if command -v micromamba >/dev/null 2>&1; then
  log_step "micromamba: $(command -v micromamba)"
else
  log_step "micromamba not found on PATH"
fi

if command -v pixi >/dev/null 2>&1; then
  log_step "pixi: $(command -v pixi)"
else
  log_step "pixi not found on PATH"
fi

if [[ -x /workspace/opt/colmap-install/bin/colmap ]]; then
  log_step "COLMAP binary: /workspace/opt/colmap-install/bin/colmap"
  /workspace/opt/colmap-install/bin/colmap -h | head -n 1 || true
else
  log_step "COLMAP binary missing or not executable: /workspace/opt/colmap-install/bin/colmap"
fi

if [[ -d /workspace/envs/splat-dev ]]; then
  log_step "Python environment exists: /workspace/envs/splat-dev"
else
  log_step "Python environment missing: /workspace/envs/splat-dev"
fi

log_section "Done"
log_step "For this shell, the Verda PATH/CUDA settings are active if you ran this script with source."
log_step "Next common step: micromamba activate /workspace/envs/splat-dev"
