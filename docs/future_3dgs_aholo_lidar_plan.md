# Future Work: LiDAR-Assisted 3DGS Publishing and Aholo Viewer

## Purpose

This document captures potential future work for extending the current
3D reconstruction pipeline into a web-oriented real-estate walkthrough
system. The main direction is to preserve the existing COLMAP +
Nerfstudio Gaussian Splatting workflow while adding metric iPhone LiDAR
data, derived collision geometry, web publishing artifacts, and an
Aholo-based browser viewer.

The intent is not to define a fixed implementation. The items below are
candidate features and architectural directions that can be selected and
developed incrementally.

## Target Experience

A future property capture could produce three complementary
representations:

1.  **Visual representation** --- a high-quality 3D Gaussian Splat
    trained with Nerfstudio.
2.  **Geometric representation** --- metric geometry derived primarily
    from iPhone LiDAR depth and suitable for collision, floor detection,
    ray queries, and navigation.
3.  **Semantic representation** --- existing room/image associations and
    room connectivity metadata, extended with reconstructed camera
    positions and optional spatial room information.

A publishing stage would convert these canonical assets into optimized
files that a browser viewer can load directly from object storage or a
CDN.

The browser application would use Aholo Viewer as the 3DGS rendering
layer while application-specific code handles real-estate navigation,
rooms, controls, annotations, loading behavior, and other product
functionality.

------------------------------------------------------------------------

## 1. Capture: iPhone 17 Pro RGB + LiDAR

A future capture workflow is expected to use an iPhone 17 Pro with LiDAR
so that RGB imagery and metric depth observations are recorded together.

For each usable capture frame, retain at least:

-   RGB image
-   LiDAR depth map
-   depth confidence information where available
-   camera intrinsics
-   device/camera metadata required to associate depth with RGB
-   timestamps
-   room identifier
-   existing visual room/link metadata

The LiDAR data should be retained as a first-class source artifact
rather than discarded after reconstruction. Its main value is providing
metric geometric observations in addition to the appearance information
used by Gaussian Splatting.

### Potential benefits

-   Recover metric scale for an otherwise scale-ambiguous COLMAP
    reconstruction.
-   Generate collision geometry independently of the Gaussian
    representation.
-   Detect bad poses or inconsistent frames using depth residuals.
-   Improve floor/wall estimation.
-   Support future measurement-related features.
-   Provide an additional quality-control signal during automated
    processing.

LiDAR depth should not be assumed to be uniformly accurate. Reflective
surfaces, mirrors, windows, difficult edges, range, incidence angle, and
low-confidence depth samples should be considered during filtering and
fusion.

------------------------------------------------------------------------

## 2. Metric Alignment and Per-Frame Quality Metrics

COLMAP should continue to provide the globally consistent camera
reconstruction. LiDAR supplies metric measurements that can be used to
align that reconstruction to real-world scale.

Rather than independently scaling every image/depth pair, a future
processing stage should estimate one robust global similarity transform
between the COLMAP coordinate system and the metric LiDAR observations.

Conceptually:

``` text
COLMAP reconstruction
        +
LiDAR depth observations
        |
        v
robust global alignment
        |
        +--> metric scale
        +--> rotation/translation alignment if required
        |
        v
metric reconstruction coordinate system
```

Once this global alignment exists, each image/depth/pose combination can
be evaluated against it.

### Candidate per-frame diagnostics

Possible metrics include:

-   estimated local scale
-   deviation from global scale
-   median absolute depth error
-   depth RMSE
-   95th percentile depth error
-   percentage of valid LiDAR depth
-   percentage of depth observations agreeing within selected thresholds
-   COLMAP reprojection/pose quality
-   LiDAR confidence statistics
-   final frame quality score

Example diagnostic artifact:

``` json
{
  "image": "IMG_01872.heic",
  "room": "kitchen",
  "local_scale": 1.0008,
  "global_scale": 1.0,
  "median_depth_error_m": 0.014,
  "rmse_depth_m": 0.027,
  "p95_depth_error_m": 0.061,
  "valid_depth_fraction": 0.71,
  "quality": 0.93
}
```

These values could later be used to reject problematic frames,
down-weight uncertain geometry, flag captures for review, or provide
property-level reconstruction quality statistics.

------------------------------------------------------------------------

## 3. Current Visual Pipeline

The existing COLMAP + Nerfstudio workflow should remain the main visual
reconstruction pipeline.

A conceptual future flow is:

``` text
RGB images
    |
    v
COLMAP
    |
    +--> camera poses
    +--> sparse reconstruction
    |
    v
Nerfstudio / Splatfacto
    |
    v
full-quality Gaussian Splat
    |
    v
master.ply
```

`master.ply` should be retained as a canonical visual asset.

Aholo-specific output should be treated as a derived publishing format
rather than the only copy of the reconstruction. This allows future
reprocessing for different renderers, compression schemes, WebGPU
implementations, or updated Aholo formats without retraining the scene.

------------------------------------------------------------------------

## 4. TSDF Fusion and Canonical Geometry

A future geometry pipeline could use the aligned RGB/LiDAR observations
and reconstructed camera poses to build a TSDF representation.

The preferred conceptual flow is:

``` text
LiDAR depth
    +
camera intrinsics
    +
metric COLMAP poses
    |
    v
depth filtering / confidence weighting
    |
    v
TSDF fusion
    |
    v
surface mesh
    |
    v
geometry cleanup
```

The TSDF stage should integrate multiple observations of the same
surfaces instead of treating individual depth frames as independent
geometry.

Potential integration weights could consider:

-   LiDAR confidence
-   distance from sensor
-   incidence angle
-   consistency with neighboring observations
-   consistency with the global reconstruction
-   per-frame pose/depth quality

The resulting geometry does not need to reproduce the visual appearance
of the property. Its purpose is to provide a stable metric surface
representation.

------------------------------------------------------------------------

## 5. Collision Geometry Variants

Several collision-generation approaches should remain available for
experimentation.

### Variant A --- TSDF-derived collision mesh

This is likely the most useful direction when good iPhone LiDAR data is
available.

``` text
LiDAR + poses
    |
    v
TSDF
    |
    v
dense surface mesh
    |
    v
cleanup / hole handling
    |
    v
simplification
    |
    v
collision mesh
```

Advantages:

-   metric
-   independent of Gaussian rendering artifacts
-   potentially good representation of floors and walls
-   reusable outside Aholo

Potential issues:

-   glass and mirrors
-   incomplete coverage
-   noisy geometry around edges
-   unwanted small objects
-   holes caused by missing depth

### Variant B --- Voxel collision representation

The TSDF or cleaned mesh could be converted to an occupancy/voxel
representation.

This may be particularly suitable for:

-   camera collision
-   wall collision
-   ground detection
-   capsule tests
-   ray queries
-   room occupancy checks

Voxel resolution could be tuned separately from visual quality.
Collision generally does not require centimeter-perfect geometry.

### Variant C --- Aholo-generated collider

Aholo's tooling appears to include voxel-related processing and
collision functionality. A future investigation should determine the
exact public/offline API and output format.

If suitable, Aholo could potentially generate its preferred collision
representation during the publishing stage.

This should be compared against the LiDAR/TSDF-derived approach rather
than automatically becoming the canonical geometry source.

### Variant D --- COLMAP / reconstructed geometry fallback

For captures without useful LiDAR, geometry could potentially be
reconstructed from COLMAP points, estimated depth, or another
surface-reconstruction method.

This would mainly serve as a fallback path and would likely be less
reliable metrically than LiDAR-assisted TSDF fusion.

### Recommended asset ownership

Keep a renderer-independent canonical geometry asset where possible:

``` text
canonical geometry
      |
      +--> simplified GLB collision mesh
      +--> occupancy voxels
      +--> Aholo collider
      +--> future engine-specific formats
```

This prevents navigation and collision data from becoming permanently
coupled to one viewer implementation.

------------------------------------------------------------------------

## 6. Aholo Publishing Stage

Aholo Viewer is a candidate browser rendering engine rather than the
complete application.

The public `@manycore/aholo-viewer` package is intended for browser
applications and supports loading 3DGS content, cameras, scenes,
rendering configuration, and related engine functionality.

Aholo also has a public `@manycore/aholo-splat-transform` component with
LOD/chunk-related functionality such as `AutoChunkLod`. A future
implementation task should verify the exact programmatic/CLI interface
and determine the most stable way to automate it in the processing
environment.

The intended publishing concept is:

``` text
master.ply
    |
    v
Aholo transform / optimization
    |
    +--> compressed splat resources
    +--> spatial chunks
    +--> LOD hierarchy
    +--> streaming metadata
    |
    v
web-ready Aholo dataset
```

### Open implementation questions

Before depending on this stage, verify:

-   exact `@manycore/aholo-splat-transform` API
-   whether a supported CLI exists
-   accepted input formats
-   output directory/manifest structure
-   AutoChunkLod configuration
-   deterministic/reproducible processing
-   CPU/GPU requirements
-   processing time for typical property scenes
-   ability to run non-interactively in a container
-   relationship between Aholo voxel tooling and runtime collision
    support
-   whether collider generation is independently usable from the
    open-source/public toolchain

The desired result is a fully unattended transformation from a master
splat to browser-ready assets.

------------------------------------------------------------------------

## 7. Room and Connectivity Metadata

Existing room information should become a core part of the published
property dataset.

Because source images already have room associations and the
reconstruction provides camera poses, room labels can be connected to
reconstructed 3D positions.

Potential derived information includes:

-   representative camera position for each room
-   representative viewing direction
-   room camera clusters
-   approximate spatial room bounds
-   room-to-room connectivity graph
-   transition/doorway estimates
-   floor association
-   preferred room entry positions
-   viewer navigation targets
-   likely next-room prefetch hints

Conceptually:

``` text
image room labels
       +
COLMAP camera poses
       +
known visual room links
       |
       v
3D room/navigation graph
```

Example:

``` json
{
  "rooms": {
    "living_room": {
      "cameraCentroid": [4.3, 1.6, -2.1],
      "neighbors": ["kitchen", "hallway"]
    },
    "hallway": {
      "cameraCentroid": [0.2, 1.6, -1.8],
      "neighbors": ["living_room", "bedroom_1", "bathroom"]
    }
  }
}
```

This metadata could eventually support both free walking and guided
Matterport-style movement.

------------------------------------------------------------------------

## 8. Room-Aware Streaming

Aholo's renderer can manage rendering/LOD based on the scene and camera,
but the application can potentially add semantic knowledge about where
the visitor is likely to move.

For example:

``` text
Current room: Living room

Living room      FULL DETAIL
Kitchen          HIGH / PREFETCHED
Hallway          HIGH / PREFETCHED
Bedroom          LOW
Bathroom         LOW
Other floor      MINIMAL / UNLOADED
```

When the user approaches the hallway, adjacent rooms could be prefetched
before they become visible.

This should be treated as a future optimization rather than an initial
requirement. First validate Aholo's native LOD and streaming behavior
with realistic property scenes.

------------------------------------------------------------------------

## 9. Proposed Automated Processing Worker

The existing VM/pod processing model can potentially be extended rather
than replaced.

A future worker image could contain the dependencies needed for:

``` text
job manifest
     |
     v
COLMAP
     |
     v
Nerfstudio
     |
     +--------------------+
     |                    |
     v                    v
 master.ply         metric alignment
                          |
                          v
                     TSDF fusion
                          |
                          v
                   canonical geometry
     |                    |
     +----------+---------+
                |
                v
          publishing stage
                |
       +--------+---------+
       |        |         |
       v        v         v
    Aholo    collision   semantic
    LOD      assets      metadata
       |        |         |
       +--------+---------+
                |
                v
          publish manifest
                |
                v
          object storage/CDN
```

The pod could continue to clone the processing repository at startup and
execute versioned processing logic in the same general manner as the
existing COLMAP/Nerfstudio jobs.

GPU resources would be used where beneficial for reconstruction and any
Aholo transformation steps that require them.

------------------------------------------------------------------------

## 10. Proposed Job Inputs

A future job could begin with a manifest plus raw capture data:

``` text
/job/
├── capture.json
├── room_metadata.json
├── images/
│   ├── ...
│   └── ...
└── depth/
    ├── ...
    └── ...
```

`capture.json` could eventually contain device calibration, timestamps,
depth associations, capture version, and processing options.

------------------------------------------------------------------------

## 11. Intermediate Artifacts

Intermediate artifacts should remain inspectable for debugging and
selective reprocessing:

``` text
/work/
├── colmap/
├── nerfstudio/
├── master.ply
├── metric_alignment.json
├── frame_quality.json
├── tsdf/
├── geometry_master.glb
├── collision/
└── processing_report.json
```

This allows individual stages to be rerun without repeating expensive
training.

Examples:

-   collider algorithm changes -\> rerun geometry/collision stages
-   Aholo format changes -\> rerun publishing stage from `master.ply`
-   room graph algorithm changes -\> rerun semantic processing
-   viewer changes -\> no reconstruction required

------------------------------------------------------------------------

## 12. Proposed Published Dataset

A property could eventually be published as a self-contained dataset:

``` text
/property-id/
├── manifest.json
├── splats/
│   ├── metadata / LOD manifest
│   └── chunks...
├── collision/
│   └── collider data
├── navigation/
│   ├── rooms.json
│   └── portals.json
├── cameras/
│   └── poses.json
├── diagnostics/
│   └── quality.json
└── preview/
    ├── hero.webp
    └── floorplan.webp
```

The exact Aholo-generated structure should be preserved where required
rather than forcing it into this example layout.

The top-level application manifest can reference those engine-specific
resources.

------------------------------------------------------------------------

## 13. Browser Viewer

The web application should treat Aholo as the rendering engine inside a
larger property-viewing product.

Responsibilities can be separated approximately as follows.

### Aholo

-   Gaussian rendering
-   GPU resource management
-   splat loading
-   LOD
-   chunk streaming
-   camera/render primitives
-   optional engine-specific collision functionality

### Application

-   property/listing UI
-   room navigation
-   walking controls
-   collision controller
-   mobile controls
-   camera presets
-   room graph
-   annotations
-   floor-plan integration
-   loading UX
-   analytics
-   permissions/access control
-   quality/device profiles
-   CDN URL management

The application should load published property assets from object
storage/CDN rather than requiring server-side rendering.

------------------------------------------------------------------------

## 14. Canonical Property Assets

A useful long-term principle is to retain three renderer-independent
canonical layers.

``` text
                 PROPERTY
                    |
       +------------+------------+
       |            |            |
       v            v            v
     VISUAL      GEOMETRIC     SEMANTIC
       |            |            |
  master.ply    TSDF/mesh     room/image
                              metadata
       |            |            |
       +------------+------------+
                    |
                    v
                PUBLISHER
                    |
                    v
          engine/web-specific assets
```

### Visual master

`master.ply` or another full-quality Gaussian representation.

### Geometric master

Metric TSDF-derived mesh and/or another stable geometry representation.

### Semantic master

Room assignments, image associations, room connectivity, capture
metadata, and any future annotations.

Aholo assets, collision encodings, previews, and optimized web
representations are then reproducible derived artifacts.

------------------------------------------------------------------------

## 15. Candidate Development Features

Potential work can be selected incrementally rather than implementing
the entire architecture at once.

### Capture and reconstruction

-   Record synchronized iPhone RGB + LiDAR depth.
-   Preserve LiDAR confidence/calibration information.
-   Add metric scale alignment to the COLMAP pipeline.
-   Generate per-frame LiDAR/COLMAP consistency metrics.
-   Add automatic bad-frame rejection.
-   Store property-level reconstruction diagnostics.

### Geometry

-   Prototype TSDF fusion from iPhone depth + COLMAP poses.
-   Export canonical metric mesh.
-   Experiment with mesh cleanup and simplification.
-   Generate collision mesh.
-   Generate voxel occupancy representation.
-   Detect likely floors/walls.
-   Compare TSDF collision against Aholo-generated voxels.

### Aholo publishing

-   Integrate `@manycore/aholo-splat-transform`.
-   Verify and automate AutoChunkLod.
-   Benchmark generated chunk/LOD sizes.
-   Determine optimal property presets.
-   Package the Aholo stage into the existing VM/pod workflow.
-   Upload generated assets to object storage/CDN.

### Semantic/navigation

-   Convert room-labeled image poses into room camera clusters.
-   Generate room centroids.
-   Generate room connectivity graph.
-   Estimate transition/portal locations.
-   Generate recommended room viewpoints.
-   Add semantic prefetch hints.

### Viewer

-   Build minimal Aholo web viewer.
-   Implement desktop orbit/free-look controls.
-   Implement first-person walkthrough controls.
-   Add collision against generated geometry.
-   Add room selection.
-   Add mobile/touch navigation.
-   Add floor-plan integration.
-   Add adaptive quality profiles.
-   Add room-aware prefetching.

------------------------------------------------------------------------

## 16. Suggested Validation Milestones

### Milestone A --- Aholo feasibility

Take one existing `master.ply`, convert it into Aholo's optimized/LOD
representation, host it from object storage, and render it in a minimal
browser application.

Measure:

-   conversion time
-   published dataset size
-   first-visible-frame time
-   bandwidth
-   FPS
-   GPU memory
-   RAM usage
-   desktop behavior
-   iPhone behavior

### Milestone B --- Metric LiDAR geometry

Capture one representative property with RGB + LiDAR.

Run:

``` text
COLMAP -> metric alignment -> TSDF -> mesh
```

Compare reconstructed dimensions and surfaces against known
measurements.

### Milestone C --- Collision walkthrough

Use the TSDF-derived geometry only for collision while Aholo renders the
Gaussian scene.

Validate:

-   floor following
-   wall collision
-   doorway traversal
-   stairs if applicable
-   camera stability
-   problematic mirrors/windows

### Milestone D --- Automated publish job

Convert the experimental scripts into a reproducible VM/pod job:

``` text
raw capture + manifest
        |
        v
one processing command
        |
        v
published property dataset
```

No manual editor or viewer-side preprocessing should be required.

### Milestone E --- Semantic property navigation

Use the existing room/image metadata to add room selection, transitions,
preferred viewpoints, and eventually room-aware streaming/prefetching.

------------------------------------------------------------------------

## 17. Key Open Questions

The following should be resolved experimentally before making them hard
architectural dependencies:

-   Exact iPhone LiDAR capture API/data format and calibration data to
    preserve.
-   Accuracy and stability of global LiDAR-to-COLMAP metric alignment.
-   Best filtering strategy for mirrors, windows, and reflective
    surfaces.
-   TSDF resolution appropriate for residential properties.
-   Whether one TSDF volume or spatially partitioned volumes are
    preferable.
-   Collision mesh versus voxel collision runtime performance.
-   Exact public interface of `@manycore/aholo-splat-transform`.
-   Exact AutoChunkLod input/output workflow.
-   Whether Aholo collider generation is suitable for unattended offline
    use.
-   Whether custom TSDF geometry can be converted directly into Aholo's
    preferred collision representation.
-   Typical Aholo dataset size after processing Nerfstudio output.
-   Mobile Safari performance on realistic full-property captures.
-   Whether room-aware prefetching materially improves native Aholo LOD
    behavior.
-   CDN request count and caching behavior for highly chunked scenes.

------------------------------------------------------------------------

## 18. General Direction

The existing pipeline does not need to be replaced. The main future
change is to evolve it from a **Gaussian training pipeline** into a
**property publishing pipeline**.

The high-level direction is:

``` text
TODAY

capture
   -> COLMAP
   -> Nerfstudio
   -> Gaussian Splat


POTENTIAL FUTURE

RGB + LiDAR + room metadata
           |
           v
        COLMAP
           |
           v
      Nerfstudio
           |
     +-----+------+
     |            |
     v            v
master splat   metric geometry
     |            |
     +------+-----+
            |
            v
      automated publisher
            |
      +-----+------+
      |            |
      v            v
 Aholo LOD      collision +
 dataset        navigation
      |            |
      +-----+------+
            |
            v
        CDN/storage
            |
            v
     real-estate viewer
```

The most important architectural goal is that all expensive
reconstruction, LOD generation, geometry processing, and semantic
preparation happen before publication. The browser should receive
already prepared resources and focus on streaming, rendering,
navigation, and interaction.

This keeps the viewer lightweight while allowing the processing pipeline
to become progressively more sophisticated without changing the basic
web delivery model.
