# Skanea RGB-D integration with the Buildvision3D pipeline

## Status and intent

This is the integration plan for bringing Skanea captures into Buildvision3D. The importer and manifest contract have now been validated with a representative 316-frame capture through R2 import and approved preprocessing. The RGB-D associations, frame ordering, room label, and capture-level camera group survive into the approved image manifest. Depth processing is not enabled yet.

The operator-led COLMAP, ARKit-prior, propagated-camera, and depth-overlay experiments are specified in the [Skanea COLMAP and ARKit pose execution plan](skanea_colmap_pose_execution_plan.md). Follow its gates in order rather than enabling all pose/depth behavior in one run.

The desired result is one hybrid visual reconstruction in which Skanea RGB frames, other room captures, and sources such as drone video can participate in COLMAP together. Skanea depth and confidence remain attached to their RGB frames and are consumed in a later geometry-processing stage.

## Architectural decision

- Keep the existing project `sources_manifest.json` as the pipeline's authoritative index of raw project inputs. Do not introduce a parallel LiDAR-only project manifest.
- Treat each Skanea RGB frame as an ordinary image source for the existing preprocessing and COLMAP path. Add stable capture/frame identity and references to its depth and confidence sidecars, along with the metadata needed to interpret them.
- Preserve Skanea's original per-session `capture.json` and raw files unchanged as capture provenance. The project manifest should register/reference these artifacts; it should not replace the app's capture record.
- Keep depth processing separate from visual matching: COLMAP initially sees RGB only. A post-COLMAP, pre-training stage uses registered RGB camera poses to filter, align, and evaluate the associated depth.
- Keep depth optional. A project without RGB-D data, or with unusable alignment, must retain the current COLMAP-only path.

The project manifest now uses schema version 4 for this extension. Each Skanea RGB source has `source_kind: skanea_rgbd_frame`, a collision-safe `preprocess_name`, and an `rgbd` attachment containing the capture/frame identity, timestamp, raw depth/confidence references, dimensions/encodings, camera intrinsics, ARKit camera-to-world transform, and coordinate descriptions. A top-level `captures` record holds capture-level provenance and derived spatial artifact paths. The original `capture.json` remains a referenced raw artifact.

## Hybrid COLMAP model

Every Skanea session should have stable, distinct source and camera-group identifiers; frame ordering and the mapping from processed image names back to original capture frames must survive preprocessing and frame selection. Drone video should remain an ordered video source group. Other room captures can be additional image or video groups.

The existing hybrid matching workflow can schedule matching within ordered groups and bridge between groups. A common project or room label alone does not ensure that groups become one reconstruction: there must be sufficient visual overlap and successful cross-group matching. The workflow should expose disconnected components and make bridge choices inspectable. Drone footage can contribute visual bridges, but does not itself provide depth unless separately instrumented.

## Proposed processing flow

```text
Skanea capture export ──┐
Other room imagery ──────┼─> one project raw area + one sources manifest
Drone video ────────────┘                  │
                                           v
                               existing image preprocessing
                                           │
                                           v
                              hybrid RGB-only COLMAP mapping
                                           │
                      registered image names ↔ Skanea frame IDs
                                           │
                                           v
                      post-COLMAP depth filtering and alignment
                                           │
                            overlay + quality report first
                                           │
                           optional geometry use in training
```

### Raw registration and staging requirements

The general browser upload remains designed for individual media files. A separate Skanea session folder input in that UI uses the same importer as `scripts/import_skanea_capture.py`. It preserves a capture under `raw/skanea/<capture-id>/`, validates every referenced file and binary sidecar size, rejects unfinished/legacy captures, path traversal, symbolic links, and unregistered files, and merges its entries into the existing `sources_manifest.json` without creating a second project manifest. The CLI remains available for automation and recovery.

Pipeline staging retains registered sidecars as immutable raw objects without treating `.f32` depth or `.u8` confidence files as images. Skanea already performs temporal sampling on the phone, so grouped preprocessing automatically retains every registered `skanea_rgbd_frame`; it may compute review metrics but does not apply FPS, quality, duplicate, or target-count filtering to those frames. Ordinary videos and non-Skanea stills keep their existing filtering behavior. Grouped preprocessing uses collision-safe input names and copies each frame's `rgbd` attachment into `image_manifest.json`, so every COLMAP image can be traced back to its raw capture/frame. Unregistered frames are never paired by basename alone.

### Post-COLMAP depth stage

For each Skanea session, use the COLMAP poses of its registered RGB frames to estimate a robust transform from that session's ARKit coordinate frame into the COLMAP reconstruction frame. Sessions have independent ARKit origins; estimate and report alignment per session (or other explicitly defined alignment unit), not by applying an assumed shared phone coordinate frame. Filter depth using validity, confidence, range, and configurable outlier/consistency checks. Preserve the raw samples and report the filtering decisions.

The first result should be a diagnostic overlay and alignment report. If the visual registration or depth alignment is insufficient, mark that session unaligned and leave the COLMAP reconstruction unchanged. Only after repeatable validation should aligned depth feed a merged initialization or another training feature.

## App-first sequencing and gates

Buildvision3D integration is deliberately deferred while the Skanea app is refined. No change to the phone's raw schema is assumed necessary: the current export is the candidate input and should be tested as-is before proposing modifications.

### Phase A — Refine and validate Skanea

Continue app-side work on capture reliability, session management, metadata completeness, and export usability. Keep the raw capture simple and inspectable. Validate that RGB, depth, confidence, intrinsics, timestamps, and poses have documented associations and that a fresh export can be inspected locally. Avoid coupling the app to Buildvision3D or adding network credentials/upload behavior at this stage.

**Gate to Phase B:** agree that the current export is stable enough for an importer; document any gaps from a representative capture rather than speculatively redesigning the format.

### Phase B — Read-only compatibility spike (complete for the first representative capture)

The dependency-free importer and schema-v4 manifest extension are implemented. A fresh 316-frame Skanea capture has been imported to R2, automatically retained through grouped preprocessing, approved, and inspected in the image grid. The approved manifest maps every selected RGB image back to its depth, confidence, intrinsics, ARKit pose, capture ID, and frame index.

**Gate to Phase C:** processed image records can be mapped deterministically back to their original Skanea frame records, and all registered raw assets survive staging.

### Phase C — Hybrid RGB mapping

Connect Skanea groups to other room sources and, when useful, drone video through the existing hybrid matching workflow. Validate source-group identity, ordering, bridge matching, and reconstruction connectivity. Depth remains unused by COLMAP in this phase.

**Gate to Phase D:** COLMAP registers the expected RGB groups and provides stable frame associations suitable for depth alignment.

### First representative COLMAP run

Use the current `first_lidar_test` project as a deliberately conventional RGB-only baseline:

- matching strategy: **Single**
- matching style: **Sequential**, with loop detection enabled
- mapper mode: **Global** (the currently verified Buildvision3D path)
- feature extractor: **SIFT**
- feature matcher: **SIFT bruteforce**
- camera model: **SIMPLE_RADIAL**
- max image size: **3200**; the 1920 × 1440 inputs remain at native size

The Skanea capture is ordered and already sampled on the phone, so the first run uses sequential matching with loop detection. Previous room-scale experiments did not show a material reconstruction improvement from exhaustive matching, while sequential matching was substantially faster and scales better to larger rooms. Exhaustive remains an operator-selectable diagnostic fallback when a sequence has unusual motion, ordering problems, or insufficient loop closure.

Before training, inspect and record:

- registered image count and ratio out of 316;
- whether COLMAP produced one connected reconstruction rather than split models;
- camera trajectory continuity and obvious folds, jumps, or mirrored sections;
- sparse-point geometry and coverage in the COLMAP viewer;
- preservation of the registered image-name to Skanea RGB-D frame association.

Do not merge LiDAR points into training initialization during this first run. Once the RGB reconstruction passes inspection, the next implementation slice estimates a robust per-capture similarity transform from corresponding COLMAP and ARKit camera poses, reports alignment residuals, and renders a filtered high-confidence depth overlay. That diagnostic gate comes before any LiDAR-assisted splat initialization.

### Phase D — Depth diagnostics

Implement per-session filtering, alignment, and overlay/reporting after COLMAP and before training. Keep this path opt-in and preserve COLMAP-only fallback. Compare alignment results across representative captures and clearly report insufficient overlap or poor fit.

### Phase E — Optional training use

Only after depth diagnostics are repeatable, evaluate optional merged initialization or later geometry-aware training. Retain source artifacts, provenance, ablation comparisons, and operator control. Do not make LiDAR required for a successful pipeline run.

## Security and operational boundaries

Start with local import on the trusted Mac-side pipeline. Do not put R2 credentials in the iOS app or expose an unauthenticated controller upload endpoint to the phone. Any later direct upload requires a separately designed authenticated, narrowly scoped upload flow (for example, short-lived project/capture-scoped upload authorization), with threat review before implementation.

## Current compatibility constraints to resolve in the spike

- The ordinary-media uploader still flattens selected files; Skanea captures must use the dedicated session-folder input or CLI importer.
- Validate the schema-v4 `rgbd` attachment with a fresh current General, Indoor Room, and Structure capture.
- Measure the cost of downloading the full raw prefix during preprocessing; sidecars are retained correctly but the stage currently syncs them before pruning/staging.
- Verify collision-safe naming with multiple Skanea sessions assigned to one location.
- Hybrid matching needs actual visual bridges; room labels and shared storage do not by themselves make a connected COLMAP model.

These are scoped integration tasks for the later phases, not reasons to alter the current app export preemptively.

## Relationship to the LiDAR processing plan

[`bring_lidar_to_the_pipeline.md`](bring_lidar_to_the_pipeline.md) describes the later filtering, alignment, diagnostics, and optional training use of depth. This document is the prerequisite integration plan: it defines how Skanea RGB-D frames enter the existing shared project and hybrid COLMAP workflow, and explicitly places that work after the app-refinement gate.
