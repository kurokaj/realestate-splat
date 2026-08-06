"""RunPod GPU options shared by controller config, UI, and worker payloads."""

from __future__ import annotations

from typing import Iterable


DEFAULT_COLMAP_GPU = "NVIDIA RTX 2000 Ada Generation"
DEFAULT_TRAINING_GPU = "NVIDIA RTX A4500"


RUNPOD_AFFORDABLE_GPU_OPTIONS = [
    {"value": "NVIDIA RTX 2000 Ada Generation", "label": "RTX 2000 Ada", "price_per_hour": 0.24, "vram_gb": 16},
    {"value": "NVIDIA RTX A4000", "label": "RTX A4000", "price_per_hour": 0.25, "vram_gb": 16},
    {"value": "NVIDIA RTX 4000 Ada Generation", "label": "RTX 4000 Ada", "price_per_hour": 0.28, "vram_gb": 20},
    {"value": "NVIDIA RTX A4500", "label": "RTX A4500", "price_per_hour": 0.25, "vram_gb": 20},
    {"value": "NVIDIA RTX A5000", "label": "RTX A5000", "price_per_hour": 0.27, "vram_gb": 24},
    {"value": "NVIDIA GeForce RTX 3090", "label": "RTX 3090", "price_per_hour": 0.50, "vram_gb": 24},
    {"value": "NVIDIA GeForce RTX 4090", "label": "RTX 4090", "price_per_hour": 0.74, "vram_gb": 24},
    {"value": "NVIDIA L4", "label": "L4", "price_per_hour": 0.49, "vram_gb": 24},
    {"value": "NVIDIA A40", "label": "A40", "price_per_hour": 0.44, "vram_gb": 48},
    {"value": "NVIDIA L40", "label": "L40", "price_per_hour": 0.82, "vram_gb": 48},
    {"value": "NVIDIA L40S", "label": "L40S", "price_per_hour": 0.99, "vram_gb": 48},
    {"value": "NVIDIA RTX 6000 Ada Generation", "label": "RTX 6000 Ada", "price_per_hour": 0.84, "vram_gb": 48},
    {"value": "NVIDIA RTX A6000", "label": "RTX A6000", "price_per_hour": 0.53, "vram_gb": 48},
]

COLMAP_GPU_OPTIONS = list(RUNPOD_AFFORDABLE_GPU_OPTIONS)
TRAINING_GPU_OPTIONS = list(RUNPOD_AFFORDABLE_GPU_OPTIONS)


GPU_NAME_ALIASES = {
    "RTX 2000 Ada": "NVIDIA RTX 2000 Ada Generation",
    "RTX A4000": "NVIDIA RTX A4000",
    "RTX 4000 Ada": "NVIDIA RTX 4000 Ada Generation",
    "RTX A4500": "NVIDIA RTX A4500",
    "RTX A5000": "NVIDIA RTX A5000",
    "RTX 3090": "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX 3090": "NVIDIA GeForce RTX 3090",
    "RTX 4090": "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 4090": "NVIDIA GeForce RTX 4090",
    "L4": "NVIDIA L4",
    "NVIDIA L4": "NVIDIA L4",
    "A40": "NVIDIA A40",
    "NVIDIA A40": "NVIDIA A40",
    "L40": "NVIDIA L40",
    "NVIDIA L40": "NVIDIA L40",
    "L40S": "NVIDIA L40S",
    "NVIDIA L40S": "NVIDIA L40S",
    "RTX 6000 Ada": "NVIDIA RTX 6000 Ada Generation",
    "RTX A6000": "NVIDIA RTX A6000",
}


def normalize_gpu_name(value: str) -> str:
    cleaned = value.strip()
    return GPU_NAME_ALIASES.get(cleaned, cleaned)


def normalize_gpu_types(values: Iterable[str]) -> list[str]:
    normalized = [normalize_gpu_name(value) for value in values if value and value.strip()]
    return normalized or [DEFAULT_COLMAP_GPU]
