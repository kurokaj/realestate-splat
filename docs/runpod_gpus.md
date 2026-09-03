# RunPod GPU Notes

Status: initial provider-selection notes for the production runtime work.

Date: 2026-07-20

These notes are based on the RunPod GPU options and prices observed while
planning the first containerized COLMAP and Gaussian splatting stages. Treat the
prices as temporary marketplace data, not fixed assumptions.

## Current target image architecture

The main RunPod COLMAP image does not need Tesla V100 support because the target
RunPod pool does not include V100.

Recommended main RunPod image:

```text
CUDA_ARCHS=75;86;89
CASPAR_ENABLED=ON
```

This covers:

```text
75  Turing / T4 fallback
86  RTX A6000, RTX A5000, A40, RTX 3090
89  L4, L40, L40S, RTX 4090, RTX 6000 Ada
```

Do not target RTX 5090 with the first CUDA 12.4 image. Treat it as a separate
future image if Blackwell support becomes useful.

If a V100-compatible image is needed later:

```text
CUDA_ARCHS=70;75;86;89
CASPAR_ENABLED=OFF
```

Caspar requires CUDA architecture 75 or newer, so it cannot be enabled in a
V100-compatible image.

## Initial cost/value ranking

Best cost/value for the current pipeline:

```text
1. RTX A5000, $0.27/hr
   Best cheap 24GB test/training option if the scene fits.

2. L4, $0.39/hr
   Efficient, 24GB, good CPU/RAM. Nice COLMAP worker.

3. A40, $0.44/hr
   Cheapest 48GB option. Very attractive for larger COLMAP/training.

4. RTX A6000, $0.49/hr
   Also excellent 48GB option. Close enough to A40 that availability can decide.

5. RTX 3090, $0.46/hr
   Fast 24GB, huge system RAM. Good if A5000/L4 unavailable.

6. RTX 6000 Ada, $0.77/hr
   Strong 48GB option, cheaper than L40/L40S in the observed pricing.

7. L40 / L40S
   Good GPUs, but less cost-efficient at the observed prices.

8. RTX 4090
   Fast, but 24GB VRAM and the observed 41GB RAM / 6 vCPU configuration make it
   less attractive for COLMAP value. More interesting for training speed if a
   scene fits in 24GB.
```

Skip for now:

```text
RTX 5090
```

Reason: the first image is CUDA 12.4 based and aimed at Ampere/Ada-era GPUs.
RTX 5090 may need a newer CUDA stack and should not complicate the first
working container.

## Practical scheduling notes

For COLMAP:

```text
Prefer: L4, RTX A5000, A40, RTX A6000
Use larger 48GB cards when image count or matching cost grows.
```

For Gaussian splatting:

```text
Prefer: A40, RTX A6000, RTX 6000 Ada, L40/L40S
Use 24GB cards for smaller scenes and development smoke tests.
```

For first smoke tests:

```text
RTX A5000 or L4 should be enough.
A40 or RTX A6000 are better for testing larger property captures.
```
