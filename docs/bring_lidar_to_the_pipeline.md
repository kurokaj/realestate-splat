# Bringing LiDAR into the Buildvision3D pipeline

> **Sequencing:** Skanea's raw export is now stable enough for the local compatibility spike. The first importer/manifest-contract slice is implemented, but depth filtering and training integration remain later phases. The prerequisite architecture and staged gates are in [Skanea RGB-D integration with the Buildvision3D pipeline](skanea_rgbd_colmap_integration_plan.md). Do not introduce a separate LiDAR-only project manifest.

The concrete UI-led reconstruction sequence is maintained in the [Skanea COLMAP and ARKit pose execution plan](skanea_colmap_pose_execution_plan.md). It defines the visual controls, position-prior comparison, propagated-camera stage, depth diagnostic, and stop gates.

## Goal

Add iPhone RGB-D/LiDAR capture as an optional, measurable input to the existing COLMAP → Nerfstudio pipeline after the app-first and compatibility phases in the integration plan. Skanea RGB frames participate in the shared hybrid COLMAP map as ordinary visual sources; their depth/confidence remain associated raw sidecars and are consumed after COLMAP. Keep the current COLMAP path intact and let the operator choose whether filtered LiDAR is used only for inspection or also for Gaussian initialization.

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

The project-level `sources_manifest.json` remains the authoritative index for all project sources; do not create a parallel LiDAR-only project manifest. The app's per-session `capture.json` remains an immutable raw provenance artifact referenced by that project manifest. Schema version 4 adds top-level Skanea capture registrations and per-RGB-source `rgbd` attachments; details and validation status are tracked in the [integration plan](skanea_rgbd_colmap_integration_plan.md).

The importer should register each RGB frame as a normal visual source and associate its raw depth data. The project manifest/import representation must provide or reference:

- RGB image path and stable frame identifier
- depth-map path and depth units
- confidence-map path, when available
- camera intrinsics and distortion information
- ARKit/device pose and coordinate convention
- timestamps and RGB/depth association
- capture/session/room identifiers
- app name, app version, device model, and capture settings

Raw files remain immutable. The project manifest is authoritative for pipeline registration; the app's capture record supplies capture-level provenance. Sidecars must be retained during staging without being sent through image preprocessing as visual media.

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

### Milestone 0 — Refine and validate Skanea (complete enough for compatibility testing)

Skanea is the intended capture app and the iPhone 17 Pro is the current target. Finish the current live-quality safeguards and acquisition controls, then begin integration with one representative real capture. Treat the current raw export as the candidate contract; do not redesign it or add phone-to-R2 upload merely for pipeline convenience. See the [integration plan](skanea_rgbd_colmap_integration_plan.md) for phase gates.

### Milestone 1 — Local compatibility spike (complete for the first representative capture)

The dependency-free importer registers RGB frames in the existing project manifest, retains associated depth/confidence/capture metadata, rejects unsafe or incomplete input, preserves collision-safe paths, and keeps sidecars through manifest-authoritative staging. A fresh 316-frame capture has completed R2 import and approved grouped preprocessing with every Skanea frame retained. The matcher now represents the capture as one ordered source group while preserving per-frame RGB-D identity.

### Milestone 2 — Hybrid COLMAP comparison

Join Skanea RGB groups with other room imagery and optional drone video through the existing hybrid matching workflow. Verify frame ordering, stable source groups, cross-group visual bridges, and connected reconstruction components. Depth is not yet used to constrain COLMAP.

### Milestone 3 — Depth diagnostics and alignment

After hybrid RGB mapping is stable, implement per-session depth filtering and robust alignment into COLMAP coordinates. Write `metric_alignment.json` and `frame_quality.json`; validate with known distances, residuals, and overlay visualizations. Mark sessions unaligned rather than forcing a poor fit.

### Milestone 4 — ARKit alignment and diagnostics first

Use ARKit poses initially only as an external alignment and diagnostic signal. Compare the ARKit trajectory against the COLMAP camera trajectory, measure residuals and drift, and use the results to identify bad frames or synchronization problems. Do not yet make ARKit poses hard constraints or required COLMAP inputs. This deliberately small first step validates the coordinate conventions and data quality before changing reconstruction behavior.

### Milestone 5 — Evaluate formal COLMAP pose priors

Once ARKit alignment and diagnostics are reliable, add an experimental path that supplies ARKit position/pose information as COLMAP initialization or weighted pose priors. Measure whether it improves convergence time, registration success, camera accuracy, or final geometry compared with the visual-only baseline. Keep prior strength configurable and retain the COLMAP-only fallback because inaccurate or over-weighted priors can reduce accuracy.

### Milestone 6 — Operator-selectable pipeline path

Add UI/API/job configuration for COLMAP-only, LiDAR overlay, or merged initialization. Keep COLMAP-only as the default until quality thresholds are demonstrated.

### Milestone 7 — Nerfstudio ablation tests

Compare COLMAP-only and merged initialization across representative rooms. Measure convergence, geometry quality, scale correctness, artifacts, and runtime. Do not auto-merge until results are consistently positive.

### Milestone 8 — Multiroom registration

Support independent room captures and register them into a building frame when overlap, markers, or other registration evidence exists. Add cross-room quality diagnostics.

### Milestone 9 — Optional automation

Define quality gates for automatic LiDAR merging. If LiDAR quality is insufficient, retain the overlay and fall back to COLMAP-only training without failing the run.

## Pose priors in COLMAP

COLMAP can use known or approximate camera information, but there are several different concepts:

- **Known intrinsics**: camera calibration can be supplied or fixed.
- **Initial poses**: approximate rotations/translations can seed reconstruction or be imported through a database/custom workflow.
- **Pose priors/constraints**: external positions can be incorporated as weighted observations in a custom or extended reconstruction workflow.
- **Bundle adjustment**: COLMAP then refines poses and 3D structure using image reprojection error, and optionally other constraints if the workflow supports them.

ARKit poses are therefore useful as initial estimates or priors, but they should not automatically be treated as exact truth. They can drift, have timing offsets, or use a different coordinate convention. A practical first version is to run normal COLMAP, align and compare the trajectories, and use the residuals to detect bad frames. A later version can inject ARKit pose priors or initial poses before/during refinement.

Pose priors can help convergence when the visual matches are weak, the scene is large, or room-to-room registration is difficult. They can reduce ambiguity and prevent obviously wrong trajectories. They do not guarantee greater accuracy: inaccurate or over-weighted ARKit poses can bias bundle adjustment and produce a visually worse reconstruction. The implementation should therefore support configurable prior weights and retain a COLMAP-only fallback.
