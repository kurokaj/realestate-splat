# Bringing LiDAR into the Buildvision3D pipeline

## Goal

Add iPhone RGB-D/LiDAR capture as an optional, measurable input to the existing COLMAP → Nerfstudio pipeline. The first production design keeps the current COLMAP path intact and lets the operator choose whether filtered LiDAR is used only for inspection or also for Gaussian initialization.

```text
Raw RGB/depth/poses
        ↓
Filtered LiDAR cloud ───────┐
                            ├─ aligned overlay
COLMAP sparse cloud ────────┘
                            ↓
                 optional merged initialization
                            ↓
                       Nerfstudio
```

The long-term possibility is to remove the operator choice and automatically merge LiDAR when its quality metrics show that it is reliable.

## Design principles

- Preserve raw capture data. Never make the filtered or merged cloud the only copy.
- Keep COLMAP and LiDAR as separate artifacts until synchronization, filtering, alignment, and quality checks pass.
- Use one robust global similarity transform per alignment unit (scale, rotation, translation). Do not independently rescale each image.
- Treat LiDAR as metric geometry and a quality-control signal first; promote it to training initialization only after validation.
- Make every decision reproducible through manifests, metrics, and explicit operator settings.
- Support room-level processing and later registration into a building-wide coordinate system.

## Canonical capture input

The importer should normalize an app's export into a capture manifest containing:

- RGB image path and stable frame identifier
- depth-map path and depth units
- confidence-map path, when available
- camera intrinsics and distortion information
- ARKit/device pose and coordinate convention
- timestamps and RGB/depth association
- capture/session/room identifiers
- app name, app version, device model, and capture settings

Raw files remain immutable. The normalized manifest is the contract used by later jobs.

## Processing stages

### 1. App import and frame synchronization

Select a phone app that exposes usable RGB, depth, confidence, intrinsics, and pose data. Convert its output to the canonical manifest. Validate timestamp/frame joins before any reconstruction.

### 2. Frame selection and LiDAR filtering

Select the RGB frames used by the COLMAP job and retrieve their corresponding depth and confidence maps. Reject invalid depth, low-confidence samples, impossible ranges, isolated outliers, and problematic reflective/glass regions according to configurable thresholds.

Write both the filtered cloud and filtering statistics. Keep raw depth available for later threshold changes.

Recommended diagnostics include valid fraction, rejected fraction, distance distribution, confidence distribution, and per-frame point counts.

### 3. COLMAP reconstruction

Keep the existing visual reconstruction as the baseline. Use sequential matching within a video and exhaustive or vocabulary-tree matching between adjacent room groups, bridge frames, and hero images. The result is a visually consistent sparse cloud and camera trajectory in COLMAP coordinates.

### 4. Metric alignment

Estimate a robust similarity transform:

```text
X_metric = s · R · X_colmap + t
```

The parameters are solved jointly from many correspondences across the alignment unit. Use robust outlier rejection, then calculate per-frame residuals as diagnostics. Do not apply a separate scale correction to individual frames.

For independently captured rooms, estimate an alignment per room first. If rooms must form a single building model, register the room coordinate systems using overlap, markers, shared geometry, or point-cloud registration and ICP.

### 5. Aligned overlay and quality report

Transform the COLMAP sparse cloud and cameras into the metric frame and render them together with the filtered LiDAR cloud. Produce per-frame and aggregate quality metrics, including depth residuals, RMSE, p95 error, valid fraction, and alignment confidence.

The first visualization should provide independent visibility controls for COLMAP and LiDAR, plus camera poses and a common scale/grid. This is a diagnostic product, not merely a presentation feature.

### 6. Optional initialization selection

Expose an operator setting for the Nerfstudio job:

- **COLMAP only**: current baseline; use the transformed or existing COLMAP sparse initialization.
- **LiDAR overlay only**: generate the aligned comparison artifact without changing training initialization.
- **Merged initialization**: create a filtered, voxel-downsampled merged cloud from aligned COLMAP and LiDAR points, preserving source/provenance metadata where possible.

The merged cloud is a derived artifact and must never overwrite either source cloud. If LiDAR is absent, the pipeline automatically falls back to COLMAP-only behavior.

### 7. Nerfstudio training

Begin with controlled comparisons: identical images, cameras, and training settings for COLMAP-only versus merged initialization. Record quality and convergence metrics. LiDAR depth as a training loss is a later custom-trainer feature, separate from point initialization.

## Artifacts

At minimum, each room/run should expose:

```text
capture.json
colmap_sparse.ply
lidar_filtered.ply
metric_alignment.json
frame_quality.json
aligned_overlay.ply or alignment_preview.html
initialization_merged.ply       # only when selected
```

Keeping separate source artifacts provides provenance, debugging, repeatability, failure isolation, and ablation testing. It also allows visualization, TSDF generation, collision geometry, and Nerfstudio to consume the representation appropriate to each task.

## Milestones

### Milestone 0 — Choose the capture app

Evaluate candidate apps against the canonical manifest. Require access to RGB, depth, confidence, intrinsics, timestamps, and poses—not merely a final PLY. Record app/device/iOS compatibility and licensing/export limitations.

### Milestone 1 — Local sandbox capture

Capture one small room and inspect the files outside Buildvision3D. Confirm depth units, coordinate handedness, pose convention, timestamp synchronization, image/depth resolution, confidence semantics, and LiDAR failure modes. Generate quick RGB, depth, confidence, and point-cloud visualizations.

### Milestone 2 — Importer and filtering

Implement the canonical manifest and an importer for the selected app. Add configurable confidence/range/outlier filtering and preserve raw and filtered outputs. Produce filtering statistics and a first filtered PLY.

### Milestone 3 — COLMAP comparison

Run the existing COLMAP pipeline on the selected RGB frames. Compare ARKit and COLMAP trajectories and verify that frame associations and camera intrinsics are correct.

### Milestone 4 — Room-level metric alignment

Implement robust similarity alignment for one room. Write `metric_alignment.json` and `frame_quality.json`. Validate with known distances and overlay visualizations.

### Milestone 5 — ARKit alignment and diagnostics first

Use ARKit poses initially only as an external alignment and diagnostic signal. Compare the ARKit trajectory against the COLMAP camera trajectory, measure residuals and drift, and use the results to identify bad frames or synchronization problems. Do not yet make ARKit poses hard constraints or required COLMAP inputs. This deliberately small first step validates the coordinate conventions and data quality before changing reconstruction behavior.

### Milestone 6 — Evaluate formal COLMAP pose priors

Once ARKit alignment and diagnostics are reliable, add an experimental path that supplies ARKit position/pose information as COLMAP initialization or weighted pose priors. Measure whether it improves convergence time, registration success, camera accuracy, or final geometry compared with the visual-only baseline. Keep prior strength configurable and retain the COLMAP-only fallback because inaccurate or over-weighted priors can reduce accuracy.

### Milestone 7 — Operator-selectable pipeline path

Add UI/API/job configuration for COLMAP-only, LiDAR overlay, or merged initialization. Keep COLMAP-only as the default until quality thresholds are demonstrated.

### Milestone 8 — Nerfstudio ablation tests

Compare COLMAP-only and merged initialization across representative rooms. Measure convergence, geometry quality, scale correctness, artifacts, and runtime. Do not auto-merge until results are consistently positive.

### Milestone 9 — Multiroom registration

Support independent room captures and register them into a building frame when overlap, markers, or other registration evidence exists. Add cross-room quality diagnostics.

### Milestone 10 — Optional automation

Define quality gates for automatic LiDAR merging. If LiDAR quality is insufficient, retain the overlay and fall back to COLMAP-only training without failing the run.

## Pose priors in COLMAP

COLMAP can use known or approximate camera information, but there are several different concepts:

- **Known intrinsics**: camera calibration can be supplied or fixed.
- **Initial poses**: approximate rotations/translations can seed reconstruction or be imported through a database/custom workflow.
- **Pose priors/constraints**: external positions can be incorporated as weighted observations in a custom or extended reconstruction workflow.
- **Bundle adjustment**: COLMAP then refines poses and 3D structure using image reprojection error, and optionally other constraints if the workflow supports them.

ARKit poses are therefore useful as initial estimates or priors, but they should not automatically be treated as exact truth. They can drift, have timing offsets, or use a different coordinate convention. A practical first version is to run normal COLMAP, align and compare the trajectories, and use the residuals to detect bad frames. A later version can inject ARKit pose priors or initial poses before/during refinement.

Pose priors can help convergence when the visual matches are weak, the scene is large, or room-to-room registration is difficult. They can reduce ambiguity and prevent obviously wrong trajectories. They do not guarantee greater accuracy: inaccurate or over-weighted ARKit poses can bias bundle adjustment and produce a visually worse reconstruction. The implementation should therefore support configurable prior weights and retain a COLMAP-only fallback.

