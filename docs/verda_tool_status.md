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

    # 3. Verify the volume exists
    lsblk -f

    # 4. Mount volume if needed
    mount -a

    # 5. Verify /workspace points to persistent storage
    df -h /workspace
    ls -l /

    # 6. Activate environment
    source ~/.bashrc
    micromamba activate /workspace/envs/splat-dev

    # 7. Optionally add COLMAP to PATH for manual debugging
    export PATH=/workspace/opt/colmap-install/bin:$PATH

    # 8. Verify tools
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

Once the manual flow is stable, create a local orchestration script that does:

    local preprocess
      ↓
    rsync selected frames to Verda
      ↓
    ssh command to run remote COLMAP
      ↓
    ssh command to run remote gsplat
      ↓
    rsync final artifacts back

Example future command:

    python scripts/run_cloud_pipeline.py \
      --run runs/house_001 \
      --host verda-a6000

This script should use SSH internally, but only after the individual remote scripts are stable.

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
