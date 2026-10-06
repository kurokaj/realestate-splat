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
| A | Visual-only Global mapper — 278/316 registered in the first run | What can the existing verified path reconstruct unaided? |
| A_hybrid | Run A plus aligned ARKit poses for visually unregistered frames | Can the strongest visual reconstruction safely retain the remaining Skanea RGB-D frames? |
| B | Visual-only Incremental mapper — 244/316 registered in the first run | What changes because of mapper choice alone? |
| C | Incremental `pose_prior_mapper` — completed and rejected | Did conservative ARKit position priors improve B? No: C registered 238/316 and retained similar floaters. |
| E | High-confidence depth overlay | Does the aligned depth agree with the visual model and across frames? |

C was compared against both controls and failed its adoption gate: it registered six fewer images than B and 40 fewer than A, while visual inspection found essentially the same floaters as B. The production UI and wrapper no longer expose the experimental pose-prior mode. Its immutable run and this result record remain for provenance. A_hybrid is a separate derived result, not another COLMAP mapper run; registered cameras retain COLMAP poses and only missing eligible cameras receive aligned ARKit poses with explicit provenance.

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

Implemented: the dependency-light alignment core and CLI pair exact image-manifest identities with COLMAP text-model cameras, estimate a deterministic robust per-capture similarity, validate the documented ARKit/OpenCV camera-axis conversion, and emit per-frame position/angular residuals plus compact run summaries. Future COLMAP runs create `analysis/arkit_alignment.json` automatically; manifests without Skanea ARKit metadata are reported as `not_applicable` without changing or blocking the generic pipeline. The COLMAP UI can backfill the diagnostic from any preserved completed run without starting a GPU Pod or modifying that model.

The selected-run trajectory viewer loads the immutable sparse model and matching alignment artifact together. It overlays the sampled sparse cloud, COLMAP trajectory, aligned full ARKit trajectory, strict hybrid trajectory, accepted/rejected correspondence links, and ARKit-only frames that COLMAP did not register. Layers and an outlier-only mode can be toggled independently, and clicking a trajectory point displays its frame identity plus positional and angular disagreement. The browser renderer uses locally vendored, pinned Three.js r186 modules; it performs no runtime CDN fetch. For a single reference capture, the sparse cloud and every trajectory are transformed through the inverse alignment into the metric, gravity-aligned ARKit frame so the floor/ceiling orientation is meaningful.

Camera direction glyphs use the COLMAP/OpenCV positive-Z optical axis and an explicit wireframe frustum: the pose point is the apex and the rectangular image plane is in front of it. Schema-v1 sparse-viewer artifacts contained the opposite COLMAP forward vector; the controller upgrades those vectors in memory while new artifacts use schema v2. This correction also removes a false near-180-degree Hybrid direction change when pose provenance switches between COLMAP and ARKit.

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

### Initial numerical validation — 2026-10-01

The new analyzer was run read-only against both preserved models before enabling propagation:

- **A / Global:** 278 correspondences, 258 robust inliers (92.81%), scale 3.653176 COLMAP units per meter, position median 0.0169 m and p95 0.0575 m, angular median 0.698° and p95 2.203°. Twenty correspondences were rejected by the initial 5%-of-trajectory-extent RANSAC threshold and require inspection in the trajectory overlay.
- **B / Incremental:** 244 correspondences, all 244 inliers, scale 3.759553 COLMAP units per meter, position median 0.0167 m and p95 0.0378 m, angular median 0.625° and p95 1.127°.

The independently reconstructed scales differ, as expected for visual SfM's arbitrary gauge, while the low position and angular residuals in both runs support the implemented transform direction and camera-axis conversion. The Run A viewer now exposes all 316 ARKit poses, 278 COLMAP poses, 258 accepted pairs, 20 rejected pairs, and 38 ARKit-only frames. This is strong evidence for continuing with A_hybrid, but operator inspection of the overlay and local checks around A's rejected anchors remain mandatory before propagating the 38 missing cameras.

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

### Run B result — 2026-10-01

- Stage run: `colmap_run_b7e49f21d4b6`
- Immutable artifact prefix: `r2://buildvision3d-pipeline/projects/first_lidar_test/colmap/runs/colmap_run_b7e49f21d4b6`
- Provider job: `vtf368paat8eqw`; repository commit `e03b7017ddf48498a9c8ee460a352d52fe381cc9`
- Runtime: COLMAP 4.0.4 in the same `cuda12.4-colmap-r2-runtime-onnx-cudnn-pycolmap-sm75-sm86-sm89-r1` image as A
- Input: the same 316 images and manifest SHA-256 `9ae0ebee9f155a4d45693520e583d1ad323f2802d5457e754c873a8b58645054`
- Registered: 244/316 images (77.22%); 72 unregistered, which is 34 fewer registered cameras than A
- Selected model: `colmap/sparse/0`; 87,075 sparse points
- Mean track length: 5.841102; mean observations per registered image: 2,084.483607
- Mean reprojection error: 0.792875 px
- End-to-end runtime observed by the operator: approximately 24 minutes
- Interpretation: B has a somewhat denser connected core but materially worse frame coverage. Keep A as the strongest visual-only operational result; use B as the controlled baseline for C.

## Step 4 — Run C: conservative ARKit position priors

### Completed result and decision — 2026-10-05

- Stage run: `colmap_run_e611dfe31e49`
- Immutable artifact prefix: `r2://buildvision3d-pipeline/projects/first_lidar_test/colmap/runs/colmap_run_e611dfe31e49`
- Provider job: `w1pof82ocj256r`; repository commit `af22b66417f71e45971fd47add4c3beab7087c54`
- Runtime: COLMAP 4.0.4 in `cuda12.4-colmap-r2-runtime-onnx-cudnn-pycolmap-sm75-sm86-sm89-r1`
- Input: the same 316 images and manifest SHA-256 `9ae0ebee9f155a4d45693520e583d1ad323f2802d5457e754c873a8b58645054`
- Mode: full fresh sequential feature extraction and matching followed by `pose_prior_mapper`
- Position prior: Conservative, isotropic 10 cm standard deviation, position-only Cartesian ARKit translation
- Registered: 238/316 images, versus B's 244 and A's 278
- Selected model: 86,674 sparse points
- Operator visual inspection: floaters were essentially the same as B; no useful geometry improvement was observed
- Decision: Gate 4 failed. Do not run Strong or Relaxed variants. Retire the active pose-prior implementation and keep this immutable run only as experiment provenance.

This result does not reject ARKit poses as downstream camera data. It rejects this particular use of position-only priors during incremental COLMAP mapping for the tested room. The accepted path remains A_hybrid, where stable Run A poses are retained and guarded missing or unstable segments use aligned full ARKit poses with explicit provenance.

## Step 5 — Build A_hybrid: Run A plus aligned ARKit cameras

### Implementation

Implemented: the UI exposes **Build A_hybrid Camera Set** for an analyzed COLMAP run. The builder produces a deterministic immutable JSON artifact under the run-specific `analyses/arkit_hybrid` prefix without modifying the source sparse model. Every frame retains image, capture, intrinsics, depth, and confidence identity; stores full camera-to-world transforms in both COLMAP/OpenCV and metric reference-ARKit conventions; and records `colmap_registered` or `arkit_propagated` provenance plus explicit replacement reasons and residuals. The stage summary stores the artifact URI, selection fingerprint, counts, and transition-validation result.

Implemented: every COLMAP↔ARKit boundary is compared with the original ARKit inter-frame motion. A boundary is marked for review when the selected motion differs by more than 10 cm or 5 degrees. The read-only Run A validation produced 316 frames (193 COLMAP and 123 ARKit), six source transitions, zero transitions requiring review, a maximum translation-step delta of 2.389 cm, and a maximum angular-step delta of 0.541 degrees. This validation did not publish the artifact; the operator creates it from the UI after approving the preview.

### B_hybrid diagnostic artifact — 2026-10-04

Before publishing A_hybrid, the workflow produced a valid derived artifact from Run B (`colmap_run_b7e49f21d4b6`). It contains all 316 frames, with 220 retained COLMAP poses and 96 ARKit replacements, four source transitions, and zero transitions requiring review. The immutable artifact is `r2://buildvision3d-pipeline/projects/first_lidar_test/colmap/analyses/arkit_hybrid/colmap_run_b7e49f21d4b6/arkit_hybrid_23c60721b3cc/hybrid_camera_set.json`. Keep it as the B_hybrid comparison; it does not modify Run B and does not replace the required A_hybrid result.

Use Run A, or a later visual reconstruction that demonstrably exceeds it, and its robust ARKit-to-COLMAP alignment to create a derived camera set:

- stable registered frames keep their refined COLMAP poses and `colmap_registered` provenance;
- unregistered and alignment-rejected Skanea frames receive transformed ARKit poses with `arkit_propagated` provenance;
- registered frames that exceed the stricter 5 cm positional or 3 degree angular diagnostic limit receive transformed ARKit poses;
- consecutive unstable runs are expanded by a five-frame guard band on both sides because the visual trajectory may begin drifting before a correspondence crosses the hard limit; isolated missing frames do not expand into stable neighbors;
- every propagated pose points back to the alignment artifact and source frame;
- frames are excluded rather than propagated when the capture-level alignment is rejected;
- initial acceptance uses capture-level alignment plus local checks against neighboring registered anchor frames;
- the derived camera set never mutates the source COLMAP model.

The first implementation exposes this selection as a purple preview trajectory and records `hybrid_pose_source` plus explicit replacement reasons per frame. These thresholds are an inspectable first-capture policy, not a permanent universal acceptance rule; revise them only from additional captured-room evidence.

Do not run ordinary bundle adjustment over all propagated cameras and assume it has validated them. With no visual tracks, there is no reprojection evidence to refine those cameras.

### Operator action

From the accepted reconstruction, select **Build A_hybrid camera set**. The UI must preview counts before creation:

- COLMAP-registered cameras;
- eligible ARKit-propagated cameras;
- excluded cameras and reasons.

After creation, inspect the combined camera trajectory and scrub through propagated RGB frames.

### Gate 5

Proceed only if propagated cameras remain temporally and spatially continuous with nearby COLMAP anchors and do not form mirrored, displaced, or abruptly rotated segments. Report visual and propagated counts separately.

## Step 6 — Run E: high-confidence depth overlay

### Implementation

Implemented for operator validation: the selected COLMAP run now exposes **Build Depth Diagnostic** after an immutable hybrid camera set exists and all source transitions pass. The operation downloads only the referenced raw depth/confidence sidecars, selects confidence value 2, rejects non-finite and out-of-range samples, scales RGB intrinsics to the depth grid, and back-projects through each frame's selected metric reference-frame pose. It does not start a GPU pod and never modifies raw capture objects.

The artifact records per-frame and per-pose-source counts, deterministic settings and provenance, multi-frame agreement-cell coverage, frame-centroid spread, a local surface-normal spread/thickness proxy, and the centroid offset where COLMAP-pose and propagated-ARKit-pose depth overlap. A separate source-aware 3 cm voxel cloud is deterministically capped at 150,000 browser points. The Three.js view exposes independent COLMAP-pose and ARKit-pose depth layers alongside COLMAP points and hybrid cameras. This implementation adds no third-party dependency: it reuses NumPy already required by ARKit alignment and the vendored Three.js viewer.

The first live A_hybrid artifact still requires operator execution and visual inspection before Gate 6 can pass. The thickness and overlap values are diagnostics, not calibrated LiDAR accuracy guarantees.

### Run E result for A_hybrid — 2026-10-04

- Source hybrid artifact: `arkit_hybrid_d523b4bdb1b2` from Run A (`colmap_run_ac7df867a9d8`)
- Depth artifact: `r2://buildvision3d-pipeline/projects/first_lidar_test/colmap/analyses/depth_diagnostic/colmap_run_ac7df867a9d8/arkit_hybrid_d523b4bdb1b2/depth_diagnostic_a2a9a46c7491/depth_diagnostic.json`
- Processed: 316/316 frames with zero skips
- Valid high-confidence depth samples: 12,325,274; stride-2 back-projected samples: 3,083,416
- Source-aware 3 cm voxels: 169,467; browser sample: 150,000 points
- Multi-frame sample coverage in 10 cm agreement cells: 99.87%
- Frame-centroid spread p95: 3.49 cm
- Local surface-normal spread/thickness proxy p95: 1.68 cm
- COLMAP-pose versus propagated-ARKit-pose centroid offset p95: 4.62 cm
- Operator visual inspection: geometry looks coherent and closely matches the known-good ARKit point cloud; no obvious source-transition splitting was reported.

Gate 6 passes provisionally for A_hybrid. The near-complete multi-frame support and centimeter-scale spread are consistent with a usable indoor LiDAR overlay, but the metrics are tied to the configured 10 cm agreement cells and are not absolute sensor-accuracy estimates. Run the identical diagnostic on B_hybrid as a low-cost comparison before choosing between the two hybrid camera sets; this requires no GPU pod.

### Run E result for B_hybrid — 2026-10-04

- Source hybrid artifact: `arkit_hybrid_23c60721b3cc` from Run B (`colmap_run_b7e49f21d4b6`)
- Depth artifact: `r2://buildvision3d-pipeline/projects/first_lidar_test/colmap/analyses/depth_diagnostic/colmap_run_b7e49f21d4b6/arkit_hybrid_23c60721b3cc/depth_diagnostic_50433d497cb6/depth_diagnostic.json`
- Processed: 316/316 frames with zero skips
- Valid high-confidence depth samples: 12,325,274; stride-2 back-projected samples: 3,083,416
- Source-aware 3 cm voxels: 164,038; browser sample: 150,000 points
- Multi-frame sample coverage in 10 cm agreement cells: 99.86%
- Frame-centroid spread p95: 3.41 cm
- Local surface-normal spread/thickness proxy p95: 1.72 cm
- COLMAP-pose versus propagated-ARKit-pose centroid offset p95: 4.69 cm
- Operator visual inspection: overall geometry is effectively the same as A_hybrid. Both visual reconstructions contain similarly bad COLMAP outliers, but at different locations.

The A/B depth comparison is a practical tie. Relative to A, B changes centroid spread by -0.08 cm, thickness proxy by +0.04 cm, source offset by +0.07 cm, and multi-frame coverage by -0.01 percentage points. These sub-millimeter, opposing changes are not meaningful evidence that either hybrid is geometrically superior. Keep A_hybrid as the primary path because Run A registered 278 visual cameras versus B's 244; retain B_hybrid as the completed control. The differently located mapper outliers reinforce the decision to preserve provenance and replace guarded unstable visual segments rather than trusting either mapper's complete trajectory unconditionally.

For each accepted Skanea camera:

1. Read the raw 256 × 192 float32 depth and uint8 confidence maps.
2. Scale the 1920 × 1440 RGB intrinsics to the depth grid.
3. Start with high-confidence, finite, physically plausible samples only.
4. Back-project samples using the camera provenance selected in A_hybrid.
5. Preserve capture ID, frame index, confidence, and camera provenance on derived statistics.
6. Produce a voxel-downsampled diagnostic overlay without altering raw depth.
7. Calculate cross-frame surface disagreement and wall/surface thickness statistics where observations overlap.

### Operator action

Open the hybrid result and select **Build depth diagnostic**. Inspect the COLMAP sparse points, camera poses, and LiDAR overlay independently and together. Check walls, corners, floor, openings, and reflective surfaces.

### Gate 6

Depth is accepted for later experiments only when scale and orientation are correct and overlapping high-confidence observations form plausible surfaces without excessive splitting or thickness. Failed sessions remain available as diagnostics but do not contribute geometry.

## Step 7 — First downstream ablation

### Implementation and smoke-test status — 2026-10-05

Implemented, awaiting the first live training smoke: the Training tab now selects a specific approved immutable COLMAP run or that run's completed A_hybrid artifact. It no longer silently consumes `colmap/current`, so the rejected Run C cannot become a training input merely because it is the newest reconstruction.

For A_hybrid, the training stage downloads the immutable camera artifact and the source Run A history prefix, verifies their run identities, and prepares all selected hybrid RGB frames. Stable frames use their retained COLMAP poses and guarded replacements use aligned ARKit poses. Both pose sources are converted from the artifact's COLMAP/OpenCV camera-to-world convention into the same Nerfstudio convention used by the visual-only path.

The first comparison deliberately keeps Run A's COLMAP-calibrated camera intrinsics and Run A's sparse points for every frame. Propagated frames resolve their camera through the manifest camera group. This isolates the effect of camera coverage and trajectory; LiDAR-assisted initialization remains the third ablation.

Preflight rejects unresolved source transitions, a mismatched base run, duplicate or missing frames, non-finite or non-rigid transforms, invalid captured intrinsics, image-size/camera-size mismatches, unresolved camera groups, inconsistent pose-source counts, or a missing sparse initialization. Training summary schema v2 records the selected camera source, prepared frame count, COLMAP/ARKit counts, hybrid fingerprint and artifact URI, base run ID, and initialization source.

The first 100-step handoff smoke tests passed:

- the visual-only Run A input prepared 278 registered cameras and all 84,052 Run A sparse points;
- training run `training_run_141c3a57a120` prepared the complete A_hybrid input with 316 cameras (193 retained COLMAP poses and 123 aligned ARKit poses) and the same 84,052 sparse points;
- both runs completed, produced a checkpoint and exported a PLY without falling back to random initialization;
- both exports retained exactly 84,052 Gaussians, as expected before Splatfacto's refinement warm-up completes, so these runs validate the handoff but do not compare reconstruction quality;
- the sparse initialization contains little wall or ceiling geometry, and A_hybrid intentionally adds cameras rather than LiDAR points.

Gate 7a, camera and sparse-initialization handoff, passes. The next controlled comparison is a 5,000-step visual-only Run A versus A_hybrid run with identical settings. Keep both in the same COLMAP-derived coordinate frame for this camera-only ablation. Introduce the metric, gravity-aligned ARKit frame together with the filtered LiDAR initialization in the third ablation; apply that coordinate transform consistently to cameras and initialization points. The camera-only splat therefore need not appear floor-level even though the diagnostic viewer does.

### 5,000-step camera-only result — 2026-10-06

The controlled camera-only comparison is complete:

- visual-only Run A used 278 cameras, 84,052 COLMAP seed points, and exported 836,857 Gaussians after 5,000 steps;
- A_hybrid training run `training_run_db11da179f56` used all 316 cameras (193 COLMAP and 123 propagated ARKit), the same 84,052 seed points, and exported 838,922 Gaussians;
- neither result retained an oversized Gaussian at export;
- A_hybrid produced only 2,065 more Gaussians (0.25%) and somewhat larger high-percentile Gaussian scales, but operator inspection found essentially the same floaters, wall structure, and overall quality;
- blank walls and the ceiling remained difficult in both results.

Gate 7b concludes that the additional/replacement A_hybrid camera poses do not materially improve this room when initialization remains the same sparse visual COLMAP cloud. Retain A_hybrid as the complete 316-frame camera scaffold for the next experiment, but do not claim a camera-only quality gain from this dataset.

The next implementation slice is the third ablation: build a deterministic, filtered high-confidence LiDAR initialization in a metric gravity-aligned ARKit frame, transform every A_hybrid camera into that same frame, color retained LiDAR samples from their source RGB frames, and preserve the COLMAP-only initialization as the control. Do not use the capped browser-viewer sample as training input.

Initial implementation added on 2026-10-06, awaiting live artifact creation: the selected A_hybrid result can now build an immutable LiDAR training initialization after its depth diagnostic has been inspected. The builder re-reads the authoritative RGB, depth, and confidence objects; keeps confidence level 2 and depths from 0.15 to 8 m; samples at stride 2; forms one 3 cm voxel grid across pose sources; requires support from at least two frames; averages real RGB colors; and writes metadata, a full PLY, and a bounded preview artifact. Training exposes a separate **A_hybrid + LiDAR initialization** source and validates the COLMAP run, hybrid fingerprint, capture identity, PLY count, and gravity-aligned coordinate frame before starting Splatfacto.

This first slice deliberately produces LiDAR-only surface seeds instead of retaining unsupported sparse COLMAP points, making the third ablation interpretable and preventing known visual floaters from being copied into the new initialization. It supports exactly one continuous ARKit capture. That capture may cover several rooms because they share one ARKit world. Independently started captures require explicit scene-assembly transforms and are rejected rather than silently merged.

The current COLMAP tab only displays initialization counts and immutable artifact URIs. Preserve the generated preview artifact for a later in-app initialization viewer. After the experiments, redesign the workflow around a scene-assembly UI rather than continuing to enlarge the COLMAP tab: captures/rooms become graph nodes, verified overlap or manual constraints become edges, and camera, sparse, LiDAR, RoomPlan, and training layers become inspectable outputs of the selected assembled scene.

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
5. Record position-prior reconstruction C as completed and rejected.
6. Keep Run A as the accepted visual reconstruction.
7. Build and inspect A_hybrid.
8. Build and inspect high-confidence depth overlay E.
9. Select the explicit A/A_hybrid training input and run the controlled smoke tests, then the downstream training ablations.

Each numbered item is a stopping point. The next implementation slice is selected only after the preceding output has been reviewed.
