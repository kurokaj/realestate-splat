#!/usr/bin/env bash
# Remove Buildvision3D test-run data from the Verda block volume.
#
# Run this on the Verda GPU instance from the repository root:
#
#   cd /workspace/repo/realestate-splat
#   bash verda/cleanup_test_runs.sh --dry-run
#   bash verda/cleanup_test_runs.sh --yes
#
# This script treats every directory under /workspace/runs as disposable test
# data. It removes COLMAP databases/models, selected frames, Nerfstudio data,
# gsplat checkpoints/exports, reports, final PLY outputs, and leftover transfer
# zips for those runs.
#
# It intentionally does NOT remove:
#
#   /workspace/repo
#   /workspace/envs
#   /workspace/opt
#
# because those contain the repository, micromamba environment, COLMAP install,
# Nerfstudio checkout, Pixi, and CUDA build caches that are expensive to rebuild.

set -euo pipefail

RUN_ROOT="/workspace/runs"
DRY_RUN=1
YES=0

usage() {
  cat <<'EOF'
Usage:
  bash verda/cleanup_test_runs.sh --dry-run
  bash verda/cleanup_test_runs.sh --yes

Options:
  --run-root PATH   Run root to clean. Default: /workspace/runs
  --dry-run         Print what would be deleted. This is the default.
  --yes             Actually delete all run data under --run-root.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --run-root requires a path." >&2
        exit 2
      fi
      RUN_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      YES=0
      shift
      ;;
    --yes)
      DRY_RUN=0
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log_section() {
  printf '\n==> %s\n' "$1"
}

show_disk() {
  if [[ -e /workspace ]]; then
    df -h /workspace || true
  else
    df -h "$(dirname "$RUN_ROOT")" || true
  fi
}

path_size() {
  local path="$1"
  du -sh "$path" 2>/dev/null | awk '{print $1}'
}

if [[ "$RUN_ROOT" != /workspace/runs && "$RUN_ROOT" != /workspace/runs/* ]]; then
  echo "ERROR: Refusing to clean outside /workspace/runs: $RUN_ROOT" >&2
  echo "Pass a run root under /workspace/runs only." >&2
  exit 2
fi

log_section "Buildvision3D Verda test-run cleanup"
echo "  - run root: ${RUN_ROOT}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  - mode: dry run"
else
  echo "  - mode: DELETE"
fi

log_section "Disk before cleanup"
show_disk

if [[ ! -e "$RUN_ROOT" ]]; then
  echo "  - Nothing to clean; run root does not exist: $RUN_ROOT"
  exit 0
fi

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "ERROR: Run root is not a directory: $RUN_ROOT" >&2
  exit 2
fi

mapfile -t RUN_DIRS < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
mapfile -t RUN_ZIPS < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 2 -type f \( -name '*_upload_bundle.zip' -o -name 'cloud_artifacts.zip' -o -name 'colmap_review.zip' \) | sort)

if [[ ${#RUN_DIRS[@]} -eq 0 && ${#RUN_ZIPS[@]} -eq 0 ]]; then
  echo "  - No run directories or transfer zips found under $RUN_ROOT."
  exit 0
fi

log_section "Run directories selected"
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "  - none"
else
  for path in "${RUN_DIRS[@]}"; do
    echo "  - $(path_size "$path")  $path"
  done
fi

log_section "Transfer zips selected"
if [[ ${#RUN_ZIPS[@]} -eq 0 ]]; then
  echo "  - none"
else
  for path in "${RUN_ZIPS[@]}"; do
    echo "  - $(path_size "$path")  $path"
  done
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log_section "Dry run complete"
  echo "Nothing was deleted. Re-run with:"
  echo "  bash verda/cleanup_test_runs.sh --yes"
  exit 0
fi

if [[ "$YES" != "1" ]]; then
  echo "ERROR: Internal argument state invalid; refusing to delete." >&2
  exit 2
fi

log_section "Deleting run directories"
for path in "${RUN_DIRS[@]}"; do
  echo "  - removing $path"
  rm -rf --one-file-system "$path"
done

log_section "Deleting leftover transfer zips"
for path in "${RUN_ZIPS[@]}"; do
  if [[ -e "$path" ]]; then
    echo "  - removing $path"
    rm -f "$path"
  fi
done

log_section "Disk after cleanup"
show_disk

log_section "Cleanup complete"
echo "  - Removed all disposable Buildvision3D run data under $RUN_ROOT."
