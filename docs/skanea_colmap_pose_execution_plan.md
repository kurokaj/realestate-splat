# Skanea COLMAP and ARKit pose execution plan

## Purpose

This is the step-by-step operator plan for the first Skanea RGB-D reconstruction experiments in Buildvision3D. The operator performs each run through the Buildvision3D UI; implementation and result inspection happen between runs. Stop at every gate and review the evidence before enabling the next experiment.

The immediate goal is not to force LiDAR into COLMAP. It is to determine, with controlled comparisons, how much visual COLMAP, ARKit position priors, aligned ARKit poses, and LiDAR depth each contribute. The first dataset is the existing `first_lidar_test` project with one 316-frame Skanea capture.

## Fixed experimental rules

- Preserve the original Skanea capture, approved image manifest, and RGB-D sidecars unchanged.
- Use the same 316 RGB images, feature extraction settings, and match database for comparable mapper runs whenever the selected COLMAP command permits it.
- Give every reconstruction a separate run ID and output prefix. Never overwrite a previous model.
- Record the exact COLMAP version and full effective command/options with every run.
- Treat the normal visual reconstruction as the control.
- Treat ARKit data as measured input with uncertainty, not ground truth.
- Mark camera provenance explicitly as `colmap_registered`, `arkit_aligned`, or `arkit_propagated`.
- Do not merge LiDAR into training initialization until camera alignment and depth-overlay gates pass.
- Do not call an ARKit-propagated frame COLMAP-registered.

## Experiment matrix

| ID | Reconstruction | Question answered |
| --- | --- | --- |
| A | Visual-only Global mapper | What can the existing verified path reconstruct unaided? |
| B | Visual-only Incremental mapper | What changes because of mapper choice alone? |
| C | Incremental pose-prior mapper | What do conservative ARKit position priors add over B? |
| D | Hybrid camera set | Can aligned ARKit poses safely retain visually unregistered frames? |
| E | High-confidence depth overlay | Does the aligned depth agree with the visual model and across frames? |

Comparing A directly with C is useful operationally but does not isolate pose-prior value because both mapper type and prior use change. The controlled prior comparison is B versus C.

## Step 0 — Freeze the test input and add run provenance

### Implementation status

Implemented: the disposable Pod lifecycle adapter uses Runpod REST API v2 at `https://api.runpod.io/v2`. The flat v1 request has been replaced with the nested v2 GPU request, explicit API-client headers and v2 error reporting are retained, and create/get/delete share the v2 client. REST API v1 is retired on November 15, 2026, so the controlled COLMAP runs no longer add dependency on the old endpoint.

The v2 migration must preserve the current operational contract: one UI-selected GPU, disposable Pod creation, status polling, termination in the cleanup path, R2 completion markers, and no exposed Pod ports. Record the API version in each run.

Implemented: every real COLMAP wrapper run publishes `run_provenance.json` in both `current/` and the immutable run-history prefix. The reconstruction report embeds the same record. It contains at least:

- project and approved-preprocess identifiers;
- input image count and an input-manifest fingerprint;
- feature extractor and matcher settings;
- mapper type and effective options;
- camera model and intrinsics source;
- exact `colmap --version` output;
- start/end timestamps, status, and output artifact paths.

The UI shows the run ID and compact runtime/provenance fields needed to distinguish A, B, and C. A failed wrapper run publishes partial provenance whenever R2 remains reachable.

### Operator action

Open `first_lidar_test`, confirm that the approved image grid contains 316 Skanea frames, and do not add or remove sources during experiments A–C.

### Gate 0

Proceed only when the approved input count is 316, every image still maps to its Skanea frame record, and the runtime COLMAP version is recorded.

## Step 1 — Run A: visual-only Global baseline

### UI settings

- Grouping strategy: **Single**
- Matching style: **Sequential**
- Sequential loop detection: **On**
- Mapper: **Global**
- Feature extractor: **SIFT**
- Feature matcher: **SIFT bruteforce**
- Camera model: **SIMPLE_RADIAL**
- Maximum image size: **3200**
- ARKit pose use: **Off**
- LiDAR/depth use: **Off**

The first implementation should seed the camera intrinsics from the Skanea metadata where the current COLMAP path supports doing so, while still allowing COLMAP to refine the permitted camera parameters. Record whether the seed was actually applied.

### Operator action

Start the matching/reconstruction job in the UI. When it finishes, open the reconstruction result and inspect the sparse model and camera trajectory.

### Evidence to record

- registered images out of 316;
- number and sizes of disconnected models;
- feature and match counts;
- median and p95 reprojection error, when available;
- runtime;
- obvious trajectory folds, jumps, mirroring, or cameras outside the room;
- which frame IDs failed to register;
- preservation of image-name to Skanea-frame identity.

### Run A result — 2026-09-30

- Stage run: `colmap_run_ac7df867a9d8`
- Immutable artifact prefix: `r2://buildvision3d-pipeline/projects/first_lidar_test/colmap/runs/colmap_run_ac7df867a9d8`
- Runtime: COLMAP 4.0.4 in `docker.io/blackjokuro/buildvision3d-colmap-gpu:cuda12.4-colmap-r2-runtime-onnx-cudnn-pycolmap-sm75-sm86-sm89-r1`
- Repository commit: `c7187178cf983c87512a563e9f832eebcada34e7`
- Input: 316 images; manifest SHA-256 `9ae0ebee9f155a4d45693520e583d1ad323f2802d5457e754c873a8b58645054`
- Registered: 278/316 images (87.97%); 38 unregistered
- Selected model: `colmap/sparse/0`; 84,052 sparse points; 497,506 observations
- Mean track length: 5.919026; mean observations per registered image: 1,789.589928
- Reported mean reprojection error: 0.000531 px. Treat this as unverified until the analyzer/parser and a distribution metric are checked; it is unusually small.
- Intrinsics source: `colmap_image_reader_estimated`; ARKit intrinsics were not seeded.
- Command durations: feature extraction 14.328 s; sequential matching 101.324 s; view-graph calibration 17.166 s; Global Mapper 254.070 s; conversion 4.049 s; analyzer 0.582 s.
- Wrapper duration: approximately 407.4 s. Controller end-to-end duration, including Pod startup, transfers, and publication: approximately 539.5 s.
- Matching: sequential with loop detection; mapper: Global; ARKit poses and LiDAR depth unused.
- Still required before Gate 1: inspect the sparse model and camera trajectory; enumerate the 38 unregistered frame IDs; confirm image-to-Skanea-frame identity; record disconnected-model evidence and reprojection-error distribution if available.

### Planned sparse-point filtering comparison

Preserve the unmodified Run A sparse model as the raw control. After the
camera-pose experiments, generate versioned derivative models with COLMAP-native
multi-view filtering and compare them in the same viewer:

- **Raw:** no post-filtering.
- **Conservative:** minimum track length 3, maximum reprojection error 2 px, minimum triangulation angle 1.5 degrees.
- **Comparison:** minimum track length 4, maximum reprojection error 2 px, minimum triangulation angle 2 degrees.

Record retained point/observation counts, per-criterion removals, camera-count
invariance, and visual changes to the below-floor and outside-wall floaters.
Do not replace the raw model. Apply the same filtering contract to candidate
mapper outputs before selecting a downstream splat-initialization model. Later,
add high-confidence LiDAR agreement as a stronger, separate geometric test.

### Gate 1

Do not reject the pipeline merely because blank-wall frames are missing. This run is the control. Stop and repair the visual pipeline only if outputs are corrupt, frame identity is lost, no useful connected model is produced, or the trajectory is clearly invalid.

## Step 2 — Add ARKit-to-COLMAP alignment diagnostics

### Implementation

For frames registered in Run A:

1. Convert the stored ARKit column-major camera-to-world transform into the explicitly documented camera convention used by the diagnostic code.
2. Pair camera records by stable Skanea capture ID and frame index, never by filename guessing.
3. Estimate a robust per-capture similarity transform from ARKit camera centers to COLMAP camera centers.
4. Apply the transform to ARKit camera orientations using the verified axis conversion.
5. Produce translation and angular residuals per frame and summary distributions.
6. Show both trajectories and rejected alignment correspondences in a diagnostic view.

Write a versioned alignment artifact containing the transform, convention metadata, correspondence count, inlier selection, residual statistics, and source run ID.

### Operator action

Open Run A and select **Analyze ARKit alignment**. Inspect the overlaid trajectories from several viewpoints and review the residual summary.

### Gate 2

Proceed only if the trajectory orientation, handedness, scale, and direction are visibly correct. A numerically small residual is not sufficient if the coordinate conversion is mirrored or axes are wrong. Record the first-run residuals as observations; do not invent a permanent acceptance threshold from one capture.

## Step 3 — Run B: visual-only Incremental control

### Implementation

Expose an **Incremental** mapper option that reuses the same approved images, camera grouping, features, and matches as Run A. Keep ARKit and depth disabled.

### UI settings

Use the same settings as Run A, changing only:

- Mapper: **Incremental**
- ARKit pose use: **Off**

### Operator action

Start Run B and inspect it using the same checklist as Run A.

### Gate 3

Confirm that Run B completed with an inspectable model and report. Keep both A and B regardless of which looks better. B is the direct control for the pose-prior mapper.

## Step 4 — Run C: conservative ARKit position priors

### Implementation

Add an experimental **Incremental + ARKit position priors** mapper mode using COLMAP's documented pose-prior path for the recorded runtime version.

- Import only camera positions as formal priors unless the runtime is proven to support another constraint correctly.
- Preserve the original full ARKit transforms for diagnostics, but do not imply that COLMAP used the rotations as constraints.
- Use a robust prior loss.
- Expose a small set of named uncertainty presets rather than an unrestricted raw-number field initially.
- Make the actual covariance/standard deviation and all COLMAP options visible in the run report.
- Do not lower visual registration inlier requirements merely to increase the registered-frame count.

The first run uses the middle, conservative uncertainty preset. Stronger or weaker presets are only run after inspecting the result.

### Operator action

Use the same matching configuration as Run B and choose **Incremental + ARKit position priors**. Select the default conservative preset, start Run C, and inspect its model and report.

### Compare B versus C

- registered-image count and exact rescued/lost frame IDs;
- connected-model count;
- reprojection error;
- camera-center residual against the aligned ARKit trajectory;
- angular residual for diagnosis, even though orientation was not a formal prior;
- trajectory continuity and loop behavior;
- sparse geometry quality;
- runtime.

### Gate 4

Adopt position priors as an available mode only if C improves registration, stability, or metric behavior without visibly degrading reprojection or geometry. Keep visual-only modes available. A truly featureless image is not expected to register from a position prior alone.

## Step 5 — Build Run D: hybrid camera set for unregistered frames

### Implementation

Use the accepted visual reconstruction and its robust ARKit-to-COLMAP alignment to create a derived camera set:

- registered frames keep their refined COLMAP poses and `colmap_registered` provenance;
- unregistered Skanea frames receive transformed ARKit poses with `arkit_propagated` provenance;
- every propagated pose points back to the alignment artifact and source frame;
- frames are excluded rather than propagated when the capture-level alignment is rejected;
- initial acceptance uses capture-level alignment plus local checks against neighboring registered anchor frames;
- the derived camera set never mutates the source COLMAP model.

Do not run ordinary bundle adjustment over all propagated cameras and assume it has validated them. With no visual tracks, there is no reprojection evidence to refine those cameras.

### Operator action

From the accepted reconstruction, select **Build hybrid camera set**. The UI must preview counts before creation:

- COLMAP-registered cameras;
- eligible ARKit-propagated cameras;
- excluded cameras and reasons.

After creation, inspect the combined camera trajectory and scrub through propagated RGB frames.

### Gate 5

Proceed only if propagated cameras remain temporally and spatially continuous with nearby COLMAP anchors and do not form mirrored, displaced, or abruptly rotated segments. Report visual and propagated counts separately.

## Step 6 — Run E: high-confidence depth overlay

### Implementation

For each accepted Skanea camera:

1. Read the raw 256 × 192 float32 depth and uint8 confidence maps.
2. Scale the 1920 × 1440 RGB intrinsics to the depth grid.
3. Start with high-confidence, finite, physically plausible samples only.
4. Back-project samples using the camera provenance selected in Run D.
5. Preserve capture ID, frame index, confidence, and camera provenance on derived statistics.
6. Produce a voxel-downsampled diagnostic overlay without altering raw depth.
7. Calculate cross-frame surface disagreement and wall/surface thickness statistics where observations overlap.

### Operator action

Open the hybrid result and select **Build depth diagnostic**. Inspect the COLMAP sparse points, camera poses, and LiDAR overlay independently and together. Check walls, corners, floor, openings, and reflective surfaces.

### Gate 6

Depth is accepted for later experiments only when scale and orientation are correct and overlapping high-confidence observations form plausible surfaces without excessive splitting or thickness. Failed sessions remain available as diagnostics but do not contribute geometry.

## Step 7 — First downstream ablation

Only after Gate 6, create separate training inputs while keeping all other settings fixed:

1. visual COLMAP cameras and normal initialization;
2. hybrid camera set with normal COLMAP initialization;
3. hybrid camera set with a filtered, voxel-downsampled LiDAR-assisted initialization.

Compare geometry, blank-wall behavior, visual quality, convergence, runtime, and failure modes. Do not enable LiDAR-assisted initialization by default from a single successful room.

## Deferred experiments

These are intentionally outside the first execution sequence:

- importing a full ARKit pose skeleton and running plain triangulation/bundle adjustment;
- custom full six-degree-of-freedom ARKit pose constraints;
- depth ICP or an RGB-D pose graph;
- TSDF fusion;
- automatic prior-strength selection;
- automatic acceptance of propagated cameras or merged LiDAR geometry;
- multi-capture and drone-to-room fusion.

The full ARKit skeleton remains a useful ablation, but it is not the default because ordinary bundle adjustment only refines cameras with visual observations and does not preserve the imported poses as soft full-pose constraints.

## Operator sequence summary

The practical UI sequence is:

1. Confirm the fixed 316-frame input and run provenance.
2. Run and inspect visual Global baseline A.
3. Analyze ARKit alignment on A.
4. Run and inspect visual Incremental control B.
5. Run and compare position-prior reconstruction C.
6. Choose the accepted visual reconstruction using the comparison evidence.
7. Build and inspect hybrid camera set D.
8. Build and inspect high-confidence depth overlay E.
9. Only then start downstream training ablations.

Each numbered item is a stopping point. The next implementation slice is selected only after the preceding output has been reviewed.
