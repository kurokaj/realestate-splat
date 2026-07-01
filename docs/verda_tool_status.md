# Verda Runtime / Block Volume Status

## Current setup

The Verda GPU instance uses a persistent **block volume** mounted at:

    /mnt/GaussianSplatVolume

The working directory is exposed through:

    /workspace

where:

    /workspace -> /mnt/GaussianSplatVolume/workspace

This means project files, environments, COLMAP binaries, datasets, and run outputs should live under `/workspace`, not directly on the VM root disk.

---

## Persistent paths

    /workspace/
      envs/
        splat-dev/              # Python virtual environment / micromamba env

      opt/
        colmap-install/         # Installed COLMAP binary + libs
          bin/
            colmap              # Main COLMAP executable with global_mapper
        nerfstudio/             # Pixi-managed Nerfstudio source checkout for Splatfacto

      repo/
        realestate-splat/       # Project repository

      data/                     # Uploaded datasets / raw inputs

      runs/                     # Per-scene processing outputs

---

## Python environment

The main Python environment is:

    /workspace/envs/splat-dev

Activate it with:

    source ~/.bashrc
    micromamba activate /workspace/envs/splat-dev

This environment contains the Python-side pipeline tools, including:

- PyTorch CUDA 12.x build
- gsplat
- pycolmap-cuda12
- OpenCV
- Open3D
- Jupyter / ipykernel
- plotting and utility libraries

The environment lives on the block volume, so it persists across GPU instances as long as the volume is attached and mounted.

---

## COLMAP

COLMAP was built from source with GPU support and global mapping support.

Installed binary:

    /workspace/opt/colmap-install/bin/colmap

Verify:

    /workspace/opt/colmap-install/bin/colmap -h
    /workspace/opt/colmap-install/bin/colmap global_mapper -h

The project scripts call this COLMAP binary by absolute path and do not rely on
`colmap` from `PATH`. For manual debugging only, you can still make `colmap`
available in the current shell:

    export PATH=/workspace/opt/colmap-install/bin:$PATH

This `PATH` change is not automatically persistent across new GPU instances unless added again or sourced from a setup script.

COLMAP option namespaces vary by build. `scripts/run_colmap.py` defaults to
`option_namespace: auto`, probes the authoritative binary's command help, and
uses newer `FeatureExtraction` / `FeatureMatching` options when supported. Use
`--option-namespace feature` or `--option-namespace sift` only when debugging a
specific binary mismatch.

For `global_mapper`, the pipeline copies `database.db` to `database_global.db`,
runs `view_graph_calibrator` on the copy, and then maps from that calibrated
database. This avoids relying on missing EXIF focal-length priors from
video-extracted frames.

---

## gsplat

gsplat is installed inside the Python environment:

    /workspace/envs/splat-dev

Verify:

    python -c "import gsplat; print('gsplat OK')"

Training scripts should run only after activating the environment.

---

## New GPU instance checklist

Each time a new GPU instance is started:

    # 1. Attach the existing Verda block volume in the UI

    # 2. SSH into the instance
    ssh root@YOUR_INSTANCE_IP

    # 3. Run project runtime init from the Verda repo checkout
    cd /workspace/repo/realestate-splat
    source verda/init_runtime.sh

The init script only orchestrates the three Verda scripts:

1. `verda/start_up_script.sh` mounts `/dev/vdb` when needed, prepares
   `/workspace`, and adds micromamba/COLMAP to `PATH`.
2. `verda/install_colmap_runtime_deps.sh` installs COLMAP/Nerfstudio runtime
   apt dependencies.
3. `verda/setup_pixi_env.sh` adds Pixi/COLMAP paths and CUDA build variables.
   It sets multi-architecture CUDA extension build targets for both RTX A6000
   / Ampere (`8.6`) and RTX 6000 Ada / Ada (`8.9`):
   `TCNN_CUDA_ARCHITECTURES=86;89` and `TORCH_CUDA_ARCH_LIST=8.6;8.9`.

If runtime apt dependencies are already installed:

    cd /workspace/repo/realestate-splat
    SKIP_APT=1 source verda/init_runtime.sh

Manual checklist equivalent:

    # Verify the volume exists
    lsblk -f

    # Verify /workspace points to persistent storage
    df -h /workspace
    ls -l /

    # Activate environment
    source ~/.bashrc
    micromamba activate /workspace/envs/splat-dev

    # Optionally add COLMAP to PATH for manual debugging
    export PATH=/workspace/opt/colmap-install/bin:$PATH

    # Verify tools
    python -c "import torch, gsplat, pycolmap; print(torch.cuda.is_available())"
    /workspace/opt/colmap-install/bin/colmap global_mapper -h

---

## Local vs cloud execution model

### Short term: SSH manually, then run scripts on the GPU

During development, the recommended workflow is:

    Local Mac:
      preprocess video
      select frames
      create capture report
      upload selected frames

    Verda GPU:
      SSH into instance
      activate env
      run COLMAP script
      run training script
      export artifacts

This is simpler and easier to debug.

Example:

    ssh root@YOUR_INSTANCE_IP
    cd /workspace/repo/realestate-splat
    source ~/.bashrc
    micromamba activate /workspace/envs/splat-dev

    python scripts/run_colmap.py --run /workspace/runs/house_001
    python scripts/run_training.py --run /workspace/runs/house_001 --config configs/training_splatfacto_dev.yaml
    python scripts/export_scene.py --run /workspace/runs/house_001 --config configs/training_splatfacto_dev.yaml

---

### Medium term: local script connects over SSH

Once the manual flow is stable, use `scripts/run_cloud_pipeline.py` from the
local Mac. It does:

    local preprocess and capture preflight gate
      ↓
    zip selected run inputs and rsync one bundle to Verda
      ↓
    ssh command to run remote COLMAP and write reconstruction report
      ↓
    ssh command to run remote gsplat training/export
      ↓
    zip final artifacts/reports/logs and rsync one bundle back

The pipeline should **fail locally** when preprocessing shows a hopeless
capture, such as zero selected frames, frame count far below the target minimum,
or coverage windows below minimum for a large part of the camera path. Every
non-fatal capture should keep a human in the loop before cloud spend starts:
the default CLI prints the local capture report path and waits for the operator
to type `yes`. Warning cases show their review reasons in that prompt. Use
`--yes-to-prompts` only for intentionally non-interactive runs.

Selected frames should be transferred as a single archive, not as many small
image files. The orchestrator creates one upload zip containing
`frames_selected/`, local capture reports, and `run_config.json`, rsyncs that
bundle to Verda, and unpacks it inside the remote run directory.

The remote COLMAP step should generate a succinct HTML report in addition to
JSON. The JSON remains the machine-readable contract; the HTML is for the
operator to decide quickly whether to continue, retry, or change COLMAP
settings. The report should include parsed `model_analyzer` metrics such as
registered images, point count, observation count, mean track length,
observations per image, and mean reprojection error. `scripts/run_colmap.py`
writes:

    reports/reconstruction_report.json
    reports/reconstruction_report.html

The orchestrator downloads those COLMAP reports and `colmap/logs/model_analyzer.log`
back to:

    runs/<run>/cloud_artifacts/colmap_review/

before starting training. It then waits for the operator to type `yes`.

Example command:

    python scripts/run_cloud_pipeline.py \
      --run runs/house_001 \
      --host verda-a6000

To run preprocessing first:

    python scripts/run_cloud_pipeline.py \
      --input-dir data/raw/house_001 \
      --out runs/house_001 \
      --profile indoor_house \
      --host verda-a6000

Use `--dry-run` for high-level verification. It prints the planned local,
rsync, and SSH commands without connecting or writing new outputs.

---

### Long term: containerized job

Final production target:

    video / frames uploaded to storage
      ↓
    GPU container starts
      ↓
    container runs full pipeline
      ↓
    final splat uploaded to hosting storage
      ↓
    viewer link generated

Do not start with this. Build it after the manual SSH pipeline works reliably.

---

## Current recommendation

Use this order:

1. Run preprocessing locally.
2. Manually SSH into Verda and run COLMAP/training/export scripts.
3. Once stable, wrap SSH + rsync into `run_cloud_pipeline.py`.
4. Once stable, move to a clean container image.

The scripts should be written so they can run directly on the GPU machine first. Later, a local script can call those same scripts over SSH.
