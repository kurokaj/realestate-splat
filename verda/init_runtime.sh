#!/usr/bin/env bash
# Initialize a fresh Verda runtime for Buildvision3D.
#
# Run this on the Verda GPU instance from the repository root:
#
#   cd /workspace/repo/realestate-splat
#   source verda/init_runtime.sh
#
# Use `source` when you want the PATH, micromamba hook, COLMAP path, Pixi path,
# and CUDA build environment variables to stay active in your current shell.
# If you only want to mount/check/install runtime dependencies, this also works:
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

log_section "Buildvision3D Verda runtime initialization"
log_step "Repository root: ${REPO_ROOT}"
log_step "Script directory: ${SCRIPT_DIR}"
log_step "Expected workspace: ${WORKSPACE}"

if [[ "$(id -u)" -ne 0 ]]; then
  log_section "Warning"
  log_step "This script is expected to run as root on the Verda instance."
  log_step "Mounting the block volume and installing apt packages may fail without root privileges."
fi

run_script "1/4 Mount block volume and expose /workspace" "${SCRIPT_DIR}/start_up_script.sh"

if [[ "${SKIP_APT:-0}" == "1" ]]; then
  log_section "2/4 Install COLMAP/Nerfstudio runtime dependencies"
  log_step "Skipping apt install because SKIP_APT=1."
else
  run_script "2/4 Install COLMAP/Nerfstudio runtime dependencies" "${SCRIPT_DIR}/install_colmap_runtime_deps.sh"
fi

log_section "3/4 Install reusable shell environment helper"
log_step "Copying ${SCRIPT_DIR}/setup_pixi_env.sh to ${SETUP_ENV_TARGET}"
cp "${SCRIPT_DIR}/setup_pixi_env.sh" "${SETUP_ENV_TARGET}"
chmod +x "${SETUP_ENV_TARGET}"

log_section "4/4 Source Verda shell environment"
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
