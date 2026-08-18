# Processing Strategy Plan

## Purpose

Build one unified COLMAP orchestration path for:

1. still images and hero images
2. one ordered video
3. one video plus hero images
4. multiple videos
5. multiple videos plus hero images

The current exhaustive, sequential, and vocabulary-tree modes remain available
as direct single-strategy fallbacks. The new work should reuse COLMAP for
feature extraction, matching, geometric verification, calibration, mapping,
and model export. Buildvision3D should provide orchestration and source-group
selection around those existing capabilities.

## Decisions

### Manual source connections first

After preprocessing, the UI will show the discovered source groups as blocks.
The user can connect blocks to describe which sources should be matched or
used as bridges.

Example blocks:

```text
[Video: living room] [Video: bedroom] [Hero: hallway] [Hero: living room]
```

Example connections:

```text
[Video: living room] -- hallway bridge --> [Video: bedroom]
[Hero: hallway] ------ relevant to ------> [Video: living room]
[Hero: hallway] ------ relevant to ------> [Video: bedroom]
```

The first version is intentionally human-directed. The user has already seen
the preprocessing timeline and source files, so they can identify whether two
videos actually share a doorway, hallway, or other visual landmark. This gives
the initial matching graph a human-checked basis instead of relying on an
opaque automatic guess.

### Meaning of a bridge source

`bridge_sources` means an explicit relationship saying that one source should
help connect another source or source group into the same reconstruction. It is
metadata for the matching plan, not a new media type and not a requirement that
the source itself be a hero image.

For example, a hallway tail captured at the end of one video and the start of
another could be represented as a bridge relationship. A hero image showing a
doorway could also be a bridge source.

The first UI does not need to expose the JSON name `bridge_sources`. It should
present understandable actions such as `Connect to`, `Use as bridge`, or
`Relevant to`. The generated plan can use a stable relationship structure
internally.

### Manual now, automatic suggestions later

The first implementation will not automatically connect every source. It will
validate and execute the connections selected by the user.

Later, the system may suggest connections using vocabulary-tree retrieval,
thumbnail similarity, source locations, or successful COLMAP matches. Suggested
connections should remain reviewable and should not silently alter the plan.

## Existing Base Structures

The implementation should build on the current artifacts:

```text
raw/sources_manifest.json
preprocess/current/image_manifest.json
preprocess/current/frames_selected/
```

The manifests already provide most of the required identity information:

- source ID
- role: video, coverage image, or hero
- location
- camera group
- image name
- source ordering for video frames
- related source information where available

The normalized processing plan should add only orchestration information. It
should not duplicate full image metadata.

## Normalized Processing Plan

Create a small `matching_plan.json` for every COLMAP run. Its shape should be
stable across all five strategies:

```json
{
  "schema_version": 1,
  "strategy": "multiple_videos_plus_heroes",
  "groups": [
    {
      "id": "video_livingroom",
      "kind": "video",
      "source_ids": ["livingroom"],
      "location": "livingroom",
      "ordered": true
    },
    {
      "id": "hero_hallway",
      "kind": "hero",
      "source_ids": ["hero_hallway"],
      "location": "hallway",
      "ordered": false
    }
  ],
  "connections": [
    {
      "from": "hero_hallway",
      "to": "video_livingroom",
      "kind": "bridge",
      "matching_style": "exhaustive"
    }
  ],
  "bridge_sources": [
    {
      "source_id": "hero_hallway",
      "connects": ["video_livingroom", "video_bedroom"]
    }
  ],
  "matching_stages": []
}
```

The plan should refer to groups and source IDs rather than storing large lists
of image names. The executor resolves groups to image names from the copied
preprocess manifest. A `bridge_sources` entry identifies the source or source
group whose images should be used to connect two or more other groups. It is a
convenience representation of connections and must resolve to explicit
matching stages before execution.

## Strategy Model

Add a high-level strategy above the existing `matcher` setting:

```text
processing_strategy = single
processing_strategy = video_plus_heroes
processing_strategy = multiple_videos
processing_strategy = multiple_videos_plus_heroes
```

For `single`, the existing matcher remains the direct choice:

```text
matcher = exhaustive
matcher = sequential
matcher = vocab_tree
```

For hybrid strategies, each generated matching stage has its own matching
style. The feature extractor and feature matcher type can initially be shared
across all stages:

```text
video sequence       -> sequential
hero to video        -> exhaustive or vocab_tree
hero to hero         -> exhaustive
video bridge         -> exhaustive or vocab_tree
```

This keeps SIFT/ALIKED compatibility validation in one place while allowing
different pair-generation styles per stage.

## Common Execution Flow

All strategies should use the same stage wrapper:

```text
download preprocess/current
  -> validate image manifest
  -> build or validate matching_plan.json
  -> extract features once
  -> execute matching stages against one database
  -> apply camera-group policy
  -> optional view-graph calibration
  -> mapper/global mapper
  -> generate reports and viewer data
  -> upload current and history artifacts to R2
```

The feature extractor must run once. Matching stages must write into the same
COLMAP database so their verified matches are available to one mapper.

The mapper, camera-group handling, sparse export, viewer payload generation,
upload-complete markers, and stage result contract should remain shared.

## Strategy Execution

### Single

Use the current path without a matching plan UI requirement:

```text
feature extraction
  -> one exhaustive/sequential/vocab-tree matcher
  -> mapper
```

This is the compatibility path and the emergency backdoor for any dataset.

### Video plus heroes

The UI creates one video group and one or more hero groups. The executor runs:

```text
sequential matching inside the video
  -> selected hero-to-video connections
  -> selected hero-to-hero connections
  -> mapper
```

Targeted exhaustive matching should be the default for a small hero set.
Vocabulary-tree matching remains available for large video groups.

### Multiple videos

The UI creates one ordered group per video and lets the user connect groups that
share visual material:

```text
sequential matching inside video A
  -> sequential matching inside video B
  -> targeted matches for the selected A/B bridge
  -> mapper
```

The bridge can be a hallway tail, doorway, shared furniture, or another visible
landmark. A connection should be treated as a hypothesis that must produce
successful verified matches; it is not enough that the user drew a line.

The same structure supports videos with different viewpoints. For example, a
ground-level walk-through and an aerial or elevated video can be processed as:

```text
ground_video: sequential matching
aerial_video: sequential matching
ground_video <-> aerial_video: selected exhaustive or vocabulary-tree matching
mapper: one combined database
```

The cross-match should normally use only overlapping candidate regions or
bridge sources rather than every frame from both videos. If the two videos
share many images and the total size is manageable, exhaustive cross-matching
is the simplest and strongest option. For larger collections, vocabulary-tree
matching is the scalable option. The mapper then receives sequential and
cross-video matches together.

### Multiple videos plus heroes

This combines all available group relationships:

```text
sequential matching inside every video
  -> hero-to-relevant-video matching
  -> hero-to-hero matching where selected
  -> cross-video bridge matching
  -> mapper
```

The UI should show this as a graph, but execution remains a linear list of
matching stages against the same database.

## COLMAP Reuse and Scoped Matching

### Capability spike result: COLMAP 4.0.4

The pinned image and `latest-dev` currently resolve to the same image digest.
The image reports:

```text
COLMAP 4.0.4
Commit 9c23f694 on 2026-04-27
CUDA enabled
```

The inspected commands provide these relevant capabilities:

- `vocab_tree_matcher` supports `VocabTreeMatching.match_list_path`
- `vocab_tree_matcher` supports the built-in default vocabulary tree
- `matches_importer` supports `match_list_path` for importing already computed
  raw or verified matches
- `sequential_matcher` does not expose an image-list or pair-list option in
  this build
- `exhaustive_matcher` does not expose a pair-list option in this build
- PyCOLMAP is not installed in the image

This means vocabulary-tree query scoping is available immediately, but exact
per-video sequential scoping and exact cross-group exhaustive pair scoping
need an orchestration adapter. The adapter must still use COLMAP's own feature
matching and geometric verification.

The most promising implementation order is:

1. Use native vocabulary-tree query lists where query-side scoping is enough.
2. Add a thin custom-pair execution path using the COLMAP image/database
   structures, while keeping matching and verification inside COLMAP.
3. Avoid treating `matches_importer` as a feature matcher; it only imports raw
   or already verified matches.

The `groups: cannot find name for group ID 109` message is a container passwd
warning and is unrelated to COLMAP functionality.

The custom layer must not implement descriptors, nearest-neighbor matching, or
geometric verification.

Use COLMAP for:

- SIFT and ALIKED feature extraction
- bruteforce and LightGlue matching
- exhaustive, sequential, and vocabulary-tree candidate generation
- geometric verification
- view-graph calibration
- incremental or global mapping

The scoped matching facility has now been checked in the pinned COLMAP image.
The remaining technical task is to implement the smallest adapter needed for
the missing exact group scoping. The preferred order is:

1. Use a native COLMAP image-list or pair-list option if available.
2. Otherwise use COLMAP's custom pair matching/import interface.
3. Only if required, add a very thin wrapper around the existing COLMAP/PyCOLMAP
   API.

The wrapper must still invoke COLMAP's matching and verification code. We are
only generating which pairs should be considered.

Every matching stage must be recorded with:

- group or connection IDs
- matching style
- feature matcher type
- candidate pair count when available
- verified match result
- duration
- log path

## UI Plan

Add a new COLMAP tab before the final run submission:

```text
Preprocess -> Matching strategy -> COLMAP inspection -> Training
```

The Matching strategy tab should contain:

- strategy selector
- discovered source/group blocks
- group metadata: role, location, image count, camera group, resolution
- connection controls
- connection kind: relevant-to or bridge
- matching style per connection
- a generated matching plan preview
- validation warnings
- estimated pair counts where possible
- `Queue COLMAP` using the reviewed plan

For `single`, the tab should show the existing direct matcher controls and an
`all files` execution path.

The UI should prevent invalid configurations:

- sequential stages require ordered video groups
- hero-only connections cannot use a video overlap assumption
- ALIKED requires an ALIKED-compatible matcher
- SIFT requires a SIFT-compatible matcher
- one connected reconstruction should warn when no bridge exists between
  disconnected video groups
- camera groups should remain visible and should not silently merge

The first version may use simple draggable or connectable blocks. The important
behavior is that the resulting connections are explicit, inspectable, and
editable before queueing.

## Artifacts and Database State

R2 should contain the reproducibility data:

```text
colmap/current/matching_plan.json
colmap/current/matching_plan_summary.json
colmap/current/matching_logs/
colmap/current/reconstruction_report.json
colmap/current/stage_result.json
```

Postgres should contain only compact values needed by the controller and UI:

- selected strategy
- group count
- connection count
- matching-stage count
- connected-component count
- registered image count
- bridge success/failure summary
- final COLMAP metrics

Detailed plans, pair lists, command logs, and reports stay in R2.

## Validation Gates

Before matching:

- every selected source resolves to images in the preprocess manifest
- every video group has stable frame ordering
- camera groups are valid
- feature extractor and matcher are compatible
- every connection has valid source groups
- the selected plan does not accidentally include duplicate image groups

After matching:

- report verified matches per stage
- report connected components
- report registration by source group
- warn if a selected bridge produced no usable verified matches
- optionally require one connected component before approval

## Implementation Phases

### Phase 1: Plan model and single-strategy compatibility

- Add normalized plan dataclasses and serialization.
- Generate a `single` plan from the current settings.
- Keep current direct matcher behavior unchanged.
- Add plan validation and dry-run output.

### Phase 2: Matching-stage executor

- Refactor command construction so feature extraction, matching stages, and
  mapping are explicit shared steps.
- Add stage logging and stage summaries.
- Confirm scoped matching support in the pinned COLMAP image.
- Test multiple matching commands against one database.

### Phase 3: Video plus heroes

- Build groups from the existing image manifest.
- Add manual hero-to-video and hero-to-hero connections.
- Execute sequential plus targeted matching.
- Report hero registration and connection success.

Implementation status:

- The pinned COLMAP source is built with same-source PyCOLMAP bindings in the
  runtime image.
- A validated JSON plan can run per-group sequential stages and targeted
  exhaustive or vocabulary-tree stages against the shared database.
- The command-line and R2 stage wrappers accept a local matching plan; the UI
  connection editor remains the next integration step.

### Phase 4: Multiple videos

- Build one ordered group per video.
- Add manual cross-video bridge connections.
- Execute per-video sequential matching and bridge stages.
- Report connected components and registration by video.

### Phase 5: Multiple videos plus heroes

- Combine video, hero, and bridge connections.
- Add plan review and estimated matching cost.
- Add one-connected-component validation option.
- Test with hallway-tail captures.

### Phase 6: UI refinement and later suggestions

- Polish the block connection UI.
- Add visual previews for source groups and bridge candidates.
- Add automatic suggestions only after manual plans are reliable.

## Test Fixtures

Use small deterministic fixtures before RunPod tests:

1. still images plus heroes using exhaustive matching
2. one ordered video using sequential matching
3. one video plus heroes with a known hero-to-video bridge
4. two short videos with overlapping hallway tails
5. two short videos plus heroes connected to both videos
6. disconnected videos that must produce a clear warning or separate models

Each fixture should verify the matching plan, command sequence, shared database,
camera groups, connected-component report, and final R2 artifact contract.

## Open Questions

- Implement and test the exact group-scoping adapter required by the pinned
  COLMAP image, without reimplementing matching or geometric verification.
- Decide whether the first UI connection control is drag-and-drop, explicit
  source selectors, or both.
- Decide whether a connection with zero verified matches blocks the run or only
  produces a warning.
- Decide whether the first hybrid version should expose vocabulary-tree
  matching per connection or keep targeted exhaustive as the only hero/bridge
  option initially.
