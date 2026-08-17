# Processing Strategies

These are the five main COLMAP processing strategies for Buildvision3D. The
selected camera groups should remain separate: for example, a phone main lens,
phone ultrawide lens, and hero camera should not be forced to share one set of
intrinsics.

## 1. Still Images and Hero Images

**Example:** A manually captured apartment image set with additional hero
images showing important views or details.

Use exhaustive matching for small and medium collections, typically up to a
few hundred images. Every image pair is considered, which gives the strongest
chance of finding connections but becomes expensive as the collection grows.

Use vocabulary-tree matching for larger still-image collections. It retrieves
visually similar candidate pairs instead of comparing every possible pair.

When the number of hero images is small, match each hero against the full
image set and match the heroes against one another. This gives them a better
chance of entering the reconstruction without requiring exhaustive matching
over the entire dataset.

## 2. One Video

**Example:** One continuous walk through an apartment, with frames extracted
in capture order.

Use sequential matching. Consecutive frames normally contain enough overlap,
so the matcher avoids the quadratic cost of comparing every frame with every
other frame.

Keep the file names in capture order and choose a frame interval that avoids
both excessive duplicate frames and large gaps. Enable loop detection when the
camera revisits an earlier room or walks around the same area.

Loop detection improves long or looping paths, but it cannot replace visual
overlap. Blank walls, motion blur, and abrupt exposure changes can still leave
parts of the sequence disconnected.

## 3. One Video Plus Hero Images

**Example:** A sequential apartment video plus several carefully composed
images from a different lens or from important room viewpoints.

Use one combined database and follow this sequence:

1. Extract features for all video and hero images.
2. Run sequential matching on the video frames.
3. Match each hero against the relevant video frames.
4. Match the hero images against one another.
5. Run one mapper over the combined database.

For a small hero set, targeted hero-to-video exhaustive matching is usually the
most predictable option. If the video contains thousands of frames,
vocabulary-tree matching can retrieve candidate video frames more cheaply.

The video supplies the continuous camera path, while the hero images add extra
viewpoints, lens coverage, or detail. The hero matches must be added to the
same database so COLMAP can register them into the video reconstruction.

## 4. Multiple Videos

**Example:** One video per room, where each room is scanned separately but the
rooms have visible overlap through a hallway or doorway.

Use sequential matching within each video, then add targeted matches between
videos at likely overlap areas. A single sequential pass over unrelated videos
would incorrectly treat the end of one video and the beginning of the next as
neighbors.

A useful capture pattern is to finish each room by walking through its doorway
and scanning a few seconds of the hallway. The next room's video should begin
with some of the same hallway landmarks before entering that room. Door frames,
corners, light fixtures, artwork, furniture, and distinctive floor patterns
make useful bridge material.

The shared hallway footage should contain several overlapping frames and slow,
stable motion. A single blank-wall frame is usually not a sufficient bridge.

## 5. Multiple Videos Plus Hero Images

**Example:** Several room videos, hero images from selected rooms, and extra
images intended to connect rooms or show important details.

This is the most complex strategy. Build one combined matching graph:

```text
video A: sequential matches
video B: sequential matches
video C: sequential matches
hero images: targeted matches to relevant videos
hallway or doorway tails: targeted cross-video matches
hero-to-hero: matches where views overlap
```

Recommended flow:

1. Extract all features once.
2. Build sequential matches independently for each video.
3. Match each hero image to the video or videos where it is relevant.
4. Match hero images to one another when they show shared scene content.
5. Add cross-video matches using hallway tails, doorways, or repeated landmarks.
6. Run one mapper over the combined database.
7. Inspect registered-image counts and connected components before training.

The hallway-tail approach is especially useful here. After scanning a room,
briefly scan out into the hallway; when scanning the next room, begin by
capturing some of the same hallway landmarks. This gives COLMAP visual evidence
for the relative placement of the separate video trajectories.

The heroes should act as bridge or coverage images, not as a reason to run
exhaustive matching over every frame. Candidate matches should be limited to
the relevant room, hallway, doorway, or shared architectural area.

## When to Use Separate Splats

Use separate reconstructions when areas have no reliable visual bridge, when
their scale or camera metadata is incompatible, or when a single matching graph
would contain too many weak or ambiguous connections.

If two videos have no visual overlap, COLMAP cannot infer their relative
translation or rotation from images alone. It may produce separate models.
Those models can be aligned manually or with external measurements, but they
are not one metrically connected reconstruction. For connected apartment
captures, one reconstruction with explicit hallway or doorway bridges is the
preferred approach.

## References

- [COLMAP feature matching](https://colmap.github.io/features.html)
- [COLMAP tutorial](https://colmap.github.io/tutorial.html)
- [COLMAP camera models](https://colmap.github.io/cameras.html)
- [COLMAP output and multiple models](https://github.com/colmap/colmap/blob/main/doc/tutorial.rst)
